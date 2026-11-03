"""ad3-nodes: Digilent Analog Discovery 3 instruments as nonoha nodes.

All nodes share a single device handle (an Analog Discovery exposes its scope,
generator, supplies and digital I/O through one handle), managed by
:class:`Ad3DeviceBlock`.
"""

import logging

from nonoha_core.node_registry import NodeRegistry

from .device_block import Ad3DeviceBlock
from .io_nodes import (
    make_factory_ad3_open_drain_node,
    make_factory_ad3_supply_node,
    make_factory_ad3_uart_rx_node,
)
from .scope_nodes import make_factory_ad3_scope_node
from .wavegen_nodes import make_factory_ad3_wavegen_node

logger = logging.getLogger(__name__)

_device_block: Ad3DeviceBlock | None = None


def initialize() -> Ad3DeviceBlock:
    """Create (once) the shared device block."""
    global _device_block
    if _device_block is None:
        _device_block = Ad3DeviceBlock()
    return _device_block


def register_nodes(registry: NodeRegistry):
    """Register Analog Discovery nodes with the engine."""
    logger.info("ad3-nodes plugin loaded")
    device_block = initialize()
    registry.register_node("Ad3Scope", make_factory_ad3_scope_node(device_block))
    registry.register_node("Ad3WaveGen", make_factory_ad3_wavegen_node(device_block))
    registry.register_node("Ad3Supply", make_factory_ad3_supply_node(device_block))
    registry.register_node("Ad3OpenDrain", make_factory_ad3_open_drain_node(device_block))
    registry.register_node("Ad3UartRx", make_factory_ad3_uart_rx_node(device_block))


def get_device_block() -> Ad3DeviceBlock:
    """Return the shared device block, creating it if needed."""
    return initialize()


async def finalize():
    """Close any device handle still open at shutdown."""
    global _device_block
    if _device_block is not None:
        _device_block.finalize()
        _device_block = None
