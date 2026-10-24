"""Tests for WebAPI and coordinator resilience against transient Modbus lag."""

# ruff: noqa: PLR2004, S101, SLF001

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.systemair.api import (
    ModbusConnectionError,
    SystemairApiClientCommunicationError,
    SystemairAuthExpiredError,
    SystemairWebApiClient,
)
from custom_components.systemair.coordinator import (
    TRANSIENT_FAILURE_THRESHOLD,
    SystemairDataUpdateCoordinator,
)
from custom_components.systemair.modbus import parameter_map


def _make_coordinator(*, is_webapi: bool = True, initial_data: dict[str, int] | None = None) -> SystemairDataUpdateCoordinator:
    """Create a bare coordinator instance for testing update and write paths."""
    coordinator = SystemairDataUpdateCoordinator.__new__(SystemairDataUpdateCoordinator)
    coordinator._is_webapi = is_webapi
    coordinator._is_homesolution = False
    coordinator.data = initial_data
    coordinator.modbus_parameters = []
    coordinator._consecutive_update_failures = 0
    coordinator._last_write_refresh = None
    coordinator.config_entry = SimpleNamespace(options={})
    coordinator.client = SimpleNamespace()
    coordinator.async_request_refresh = AsyncMock()
    coordinator.async_set_updated_data = Mock()
    return coordinator


class WebApiClientResilienceTest(unittest.IsolatedAsyncioTestCase):
    """Verify WebAPI register batching, pacing, and partial-chunk recovery."""

    async def test_async_get_data_sorts_and_deduplicates_registers(self) -> None:
        """Registers are queried in ascending address order without duplicates."""
        client = SystemairWebApiClient(address="192.168.1.50", session=Mock(), max_registers_per_request=70)
        urls: list[str] = []

        async def fake_api_wrapper(*, method: str, url: str, expects_data: bool = False) -> dict[str, int]:
            assert method == "get"
            assert expects_data is True
            urls.append(url)
            return {"1130": 1, "2000": 200, "12101": 220}

        client._api_wrapper = fake_api_wrapper  # type: ignore[method-assign]

        reg_high = parameter_map["REG_SENSOR_SAT"]
        reg_low = parameter_map["REG_FUNCTION_ACTIVE_HEATER"]
        reg_mid = parameter_map["REG_TC_SP"]

        result = await client.async_get_data([reg_high, reg_low, reg_mid, reg_low])

        expected_addrs = sorted({reg_high.register - 1, reg_low.register - 1, reg_mid.register - 1})
        expected_query = ",".join(f"%22{addr}%22:1" for addr in expected_addrs)
        assert urls == [f"http://192.168.1.50/mread?{{{expected_query}}}"]
        assert result == {"1130": 1, "2000": 200, "12101": 220}

    async def test_async_get_data_paces_multiple_chunks_and_keeps_partial_data(self) -> None:
        """A transient failure in chunk 2 still returns data read from chunk 1."""
        client = SystemairWebApiClient(address="192.168.1.50", session=Mock(), max_registers_per_request=2)
        regs = [
            parameter_map["REG_FUNCTION_ACTIVE_HEATER"],
            parameter_map["REG_TC_SP"],
            parameter_map["REG_SENSOR_SAT"],
        ]
        responses: list[dict[str, int] | Exception] = [
            {"1130": 1, "2000": 210},
            SystemairApiClientCommunicationError("MB DISCONNECTED"),
        ]

        async def fake_api_wrapper(**_kwargs: object) -> dict[str, int]:
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        client._api_wrapper = fake_api_wrapper  # type: ignore[method-assign]

        with (
            patch("custom_components.systemair.api.asyncio.sleep", AsyncMock()) as sleep_mock,
            self.assertLogs("custom_components.systemair", level="WARNING"),
        ):
            result = await client.async_get_data(regs)

        sleep_mock.assert_awaited_once()
        assert result == {"1130": 1, "2000": 210}

    async def test_async_get_data_raises_when_all_chunks_fail(self) -> None:
        """When every chunk fails, the communication error is raised to the coordinator."""
        client = SystemairWebApiClient(address="192.168.1.50", session=Mock(), max_registers_per_request=1)
        regs = [parameter_map["REG_TC_SP"], parameter_map["REG_SENSOR_SAT"]]
        client._api_wrapper = AsyncMock(side_effect=SystemairApiClientCommunicationError("MB DISCONNECTED"))

        with (
            patch("custom_components.systemair.api.asyncio.sleep", AsyncMock()),
            self.assertLogs("custom_components.systemair", level="WARNING"),
            self.assertRaises(SystemairApiClientCommunicationError),  # noqa: PT027
        ):
            await client.async_get_data(regs)

    async def test_async_get_data_propagates_auth_errors_immediately(self) -> None:
        """Authentication errors are never swallowed as partial chunk reads."""
        client = SystemairWebApiClient(address="192.168.1.50", session=Mock(), max_registers_per_request=1)
        regs = [parameter_map["REG_TC_SP"], parameter_map["REG_SENSOR_SAT"]]
        client._api_wrapper = AsyncMock(
            side_effect=[
                {"2000": 210},
                SystemairAuthExpiredError("Auth expired"),
            ]
        )

        with (
            patch("custom_components.systemair.api.asyncio.sleep", AsyncMock()),
            self.assertRaises(SystemairAuthExpiredError),  # noqa: PT027
        ):
            await client.async_get_data(regs)


class CoordinatorResilienceTest(unittest.IsolatedAsyncioTestCase):
    """Verify coordinator grace period, partial merge, and write refresh coalescing."""

    async def test_transient_failures_return_cached_data_until_threshold(self) -> None:
        """Cached data is preserved across transient MB DISCONNECTED errors up to threshold."""
        cached = {"2000": 210}
        coordinator = _make_coordinator(initial_data=cached)
        coordinator.modbus_parameters = [parameter_map["REG_TC_SP"]]
        coordinator.client.async_get_data = AsyncMock(side_effect=SystemairApiClientCommunicationError("MB DISCONNECTED"))

        for attempt in range(1, TRANSIENT_FAILURE_THRESHOLD + 1):
            with self.assertLogs("custom_components.systemair", level="WARNING"):
                data = await coordinator._async_update_data()
            assert data == cached
            assert coordinator._consecutive_update_failures == attempt

        with self.assertRaises(UpdateFailed):  # noqa: PT027
            await coordinator._async_update_data()

        coordinator.client.async_get_data = AsyncMock(return_value={"2000": 220})
        recovered = await coordinator._async_update_data()
        assert recovered == {"2000": 220}
        assert coordinator._consecutive_update_failures == 0

    async def test_first_refresh_without_cached_data_raises_immediately(self) -> None:
        """Initial setup still fails fast when no cached data exists."""
        coordinator = _make_coordinator(initial_data=None)
        coordinator.modbus_parameters = [parameter_map["REG_TC_SP"]]
        coordinator.client.async_get_data = AsyncMock(side_effect=ModbusConnectionError("MB DISCONNECTED"))

        with self.assertRaises(UpdateFailed):  # noqa: PT027
            await coordinator._async_update_data()

    async def test_partial_poll_merges_with_previous_register_cache(self) -> None:
        """Registers from a skipped chunk retain their previous values on merge."""
        coordinator = _make_coordinator(initial_data={"1130": 1, "2000": 210})
        coordinator.modbus_parameters = [parameter_map["REG_TC_SP"]]
        coordinator.client.async_get_data = AsyncMock(return_value={"2000": 225})

        merged = await coordinator._async_update_data()
        assert merged == {"1130": 1, "2000": 225}

    async def test_auth_error_bypasses_transient_failure_tolerance(self) -> None:
        """Expired authentication raises ConfigEntryAuthFailed even when cache exists."""
        coordinator = _make_coordinator(initial_data={"2000": 210})
        coordinator.modbus_parameters = [parameter_map["REG_TC_SP"]]
        coordinator.client.async_get_data = AsyncMock(side_effect=SystemairAuthExpiredError("Auth expired"))

        with self.assertRaises(ConfigEntryAuthFailed):  # noqa: PT027
            await coordinator._async_update_data()

    async def test_write_updates_cache_and_coalesces_duplicate_refresh(self) -> None:
        """Writing a register updates self.data and triggers only one refresh within cooldown."""
        coordinator = _make_coordinator(initial_data={"2000": 200})
        coordinator.client.write_register = AsyncMock()
        reg = parameter_map["REG_TC_SP"]

        await coordinator.set_modbus_data(reg, 22)
        await coordinator.async_refresh_after_write()

        assert coordinator.data[str(reg.register - 1)] == 220
        coordinator.async_request_refresh.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
