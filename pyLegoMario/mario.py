"""
MARIO.PY
###################################################################################
MIT License
Copyright (c) 2022 Bruno Hautzenberger, Jamin Kauf
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import asyncio
from typing import Any, Callable, Iterable, Union
from bleak import BleakScanner, BleakClient, BleakError

try:
    from .lego_mario_data import (
        HEX_TO_RGB_TILE,
        HEX_TO_COLOR_TILE,
        HEX_TO_PANTS,
        HEX_TO_HUB_ACTIONS,
        HEX_TO_HUB_PROPERTIES,
        BINARY_GESTURES,
        LEGO_CHARACTERISTIC_UUID,
        SUBSCRIBE_IMU_COMMAND,
        SUBSCRIBE_PANTS_COMMAND,
        SUBSCRIBE_RGB_COMMAND,
        DISCONNECT_COMMAND,
        pifs_command,
        TURN_OFF_COMMAND,
        MUTE_COMMAND,
        REQUEST_RGB_COMMAND,
    )
except ImportError:
    from lego_mario_data import (
        HEX_TO_RGB_TILE,
        HEX_TO_COLOR_TILE,
        HEX_TO_PANTS,
        HEX_TO_HUB_ACTIONS,
        HEX_TO_HUB_PROPERTIES,
        BINARY_GESTURES,
        LEGO_CHARACTERISTIC_UUID,
        SUBSCRIBE_IMU_COMMAND,
        SUBSCRIBE_PANTS_COMMAND,
        SUBSCRIBE_RGB_COMMAND,
        DISCONNECT_COMMAND,
        pifs_command,
        TURN_OFF_COMMAND,
        MUTE_COMMAND,
        REQUEST_RGB_COMMAND,
    )


class Mario:
    """Object to control and monitor a Lego Mario via Bluetooth.

    Attributes
    ----------
    do_log: bool
        If True, log messages will be printed to stdout.
    pants: str | None
        Value of most recent pants value
    ground: str | None
        Value of most recent camera/rgb value
    acceleration: tuple[int, int, int] | None
        Value of most recent acceleration value
    auto_reconnect: bool
        Whether .connect() should be called after disconnecting
    run: bool
        TBD
    client: BleakClient | None
        The bluetooth client that communicates with Mario
    default_volume: int | None
        The % volume that Mario will be set to after (re)connecting.
    _accelerometer_hooks: list[(Mario, int, int, int) -> None]
        List of callback functions for accelerometer updates.
    _pants_event_hooks: list[(Mario, str) -> None]
        List of callback functions for pants updates.
    _tile_event_hooks: list[(Mario, str) -> None]
        List of callback functions for camera/rgb updates.
    _log_event_hooks: list[(Mario, str) -> None]
        List of callback functions for log messages.
    _all_hooks: tuple[list[callbacks]]
        tuple that contains all of the previous lists of callback functions

    Methods
    -------
    add_accelerometer_hooks: (Callable | list[Callable]) -> None
        Adds the given function(s) as callback functions for accelerometer data
    add_pants_hooks: (Callable | list[Callable]) -> None
        Adds the given function(s) as callback functions for pants data
    add_tile_hooks: (Callable | list[Callable]) -> None
        Adds the given function(s) as callback functions for tile data
    add_log_hooks: (Callable | list[Callable]) -> None
        Adds the given function(s) as callback functions for log calls
    remove_hooks: (list[Any] | Callable) -> None
        Removes the given object(s) from all hook lists.
    log: (str) -> None
        logs the message to stdout if self.do_log is true. Also passes message
        to all callback functions in self._log_hooks.
    connect: () -> Coroutine
        Searches for Lego Mario objects via bluetooth and tries to connect.
        Needs to be awaited.
    set_volume: (int) -> None
        If mario is connected, sets volume to the % volume given. Also sets
        self.default_volume to keep volume persistent with reconnects.
    """

    def __init__(
        self,
        do_log: bool = True,
        accelerometer_hooks: Union[
            Callable[["Mario", int, int, int], Any],
            Iterable[Callable[["Mario", int, int, int], Any]],
        ] = [],
        tile_event_hooks: Union[
            Callable[["Mario", str], Any], Iterable[Callable[["Mario", str], Any]]
        ] = [],
        pants_event_hooks: Union[
            Callable[["Mario", str], Any], Iterable[Callable[["Mario", str], Any]]
        ] = [],
        log_event_hooks: Union[
            Callable[["Mario", str], Any], Iterable[Callable[["Mario", str], Any]]
        ] = [],
        default_volume: Union[int, None] = None,
    ) -> None:
        """
        Args:
            do_log (bool, optional): Enables Logs to Stdout. Defaults to True.

            accelerometerEventHooks (func or list of functions, optional):
                Event Hook(s) that should be called every time new
                accelerometer data is received.
                Functions need to take four inputs
                (sender: Mario, x: int, y: int, z: int). Defaults to None.

            tileEventHooks (func or list of functions, optional):
                Event Hook(s) that should be called every time new camera
                data is received. Functions need to take two inputs:
                (sender: Mario, ground: str). Defaults to None.

            pantsEventHooks (func or list of functions, optional):
                Event hook(s) that will be called every time new pants data
                is received. Functions need to take two inputs:
                (sender: Mario, pants: str). Defaults to None.

            logEventHooks (func or list of functions, optional): Event Hook(s)
                that will be called every time something gets logged. Functions
                need to take two inputs: (sender: Mario, msg: str).
                Defaults to None.

            defaultVolume (func or list of functions, optional): Volume (0-100)
                that will be set every time Mario reconnects. If not provided,
                will not adjust volume. Defaults to None.
        """

        self.do_log = do_log  # output logs to stdout if True
        self.run = False
        self.auto_reconnect = True  # handles reconnection on disconnect
        self.client: BleakClient | None = None
        self.default_volume = default_volume  # if None, volume won't be changed

        # values to keep most recent event in memory
        self.pants: str | None = None
        self.ground: str | None = None
        self.acceleration: tuple[int, int, int] | None = None
        self.recent_tile: str | None = None

        self._accelerometer_hooks: list[Callable[[Mario, int, int, int], Any]] = []
        self._tile_event_hooks: list[Callable[[Mario, str], Any]] = []
        self._pants_event_hooks: list[Callable[[Mario, str], Any]] = []
        self._log_event_hooks: list[Callable[[Mario, str], Any]] = []
        self._all_hooks = (
            self._accelerometer_hooks,
            self._pants_event_hooks,
            self._tile_event_hooks,
            self._log_event_hooks,
        )

        self.add_accelerometer_hooks(accelerometer_hooks)
        self.add_tile_hooks(tile_event_hooks)
        self.add_pants_hooks(pants_event_hooks)
        self.add_log_hooks(log_event_hooks)

        try:  # if event loop exists, use that one
            asyncio.get_event_loop().create_task(self.connect())
        except RuntimeError:  # otherwise, create a new one
            asyncio.set_event_loop(asyncio.SelectorEventLoop())
            asyncio.get_event_loop().create_task(self.connect())

    def log(self, msg: str, end: str = "\n") -> None:
        """Log any message to stdout and call all assigned LogEvent handlers.

        Args:
            msg (object): Any printable object.
            end (str, optional): Same as end in print(). Defaults to "\n".
        """
        for func in self._log_event_hooks:
            func(self, msg)
        if self.do_log:
            address = "Not Connected" if not self.client else self.client.address
            print((f"\r{address}: {msg}").ljust(100), end=end)

    def add_log_hooks(
        self,
        funcs: Union[
            Callable[["Mario", str], Any], Iterable[Callable[["Mario", str], Any]]
        ],
    ) -> None:
        """Adds function(s) as event hooks for updated tile or color values.

        Args:
            funcs (func or list of functions): callback functions must take
                (Mario, str) as input.
        """
        if callable(funcs):
            self._log_event_hooks.append(funcs)
        elif hasattr(funcs, "__iter__"):
            for hook_function in funcs:
                self.add_log_hooks(hook_function)

    def add_tile_hooks(
        self,
        funcs: Union[
            Callable[["Mario", str], Any], Iterable[Callable[["Mario", str], Any]]
        ],
    ) -> None:
        """Adds function(s) as event hooks for updated tile or color values.

        Args:
            funcs (func or list of functions): callback functions must take
                (Mario, str) as input.
        """
        if callable(funcs):
            self._tile_event_hooks.append(funcs)
        elif hasattr(funcs, "__iter__"):
            for hook_function in funcs:
                self.add_tile_hooks(hook_function)

    def add_accelerometer_hooks(
        self,
        funcs: Union[
            Callable[["Mario", int, int, int], Any],
            Iterable[Callable[["Mario", int, int, int], Any]],
        ],
    ) -> None:
        """Adds function(s) as event hooks for updated accelerometer values.

        Args:
            funcs (function or list of functions): callback function(s) take
            input as (Mario, int, int, int).
        """
        if callable(funcs):
            self._accelerometer_hooks.append(funcs)
        elif hasattr(funcs, "__iter__"):
            for hook_function in funcs:
                self.add_accelerometer_hooks(hook_function)

    def add_pants_hooks(
        self,
        funcs: Union[
            Callable[["Mario", str], Any], Iterable[Callable[["Mario", str], Any]]
        ],
    ) -> None:
        """Adds function(s) as event hooks for updated pants values.

        Args:
            funcs (func or list of functions): callback function(s) take
                input as (Mario, str).
        """
        if callable(funcs):
            self._pants_event_hooks.append(funcs)
        elif hasattr(funcs, "__iter__"):
            for hook_function in funcs:
                self.add_pants_hooks(hook_function)

    def remove_hooks(
        self, funcs: Union[Callable[..., Any], Iterable[Callable[..., Any]]]
    ) -> None:
        """Removes function(s) as event hooks.
            Note that this is without consideration for the type of hook.

        Args:
            funcs (Union[ Callable[[Any], Any], Iterable[Callable[[Any], Any]]]):
                callable or iterable of callable.
        """
        if callable(funcs):
            for hooktype in self._all_hooks:
                if funcs in hooktype:
                    hooktype.remove(funcs)
        elif hasattr(funcs, "__iter__"):
            for hook_function in funcs:
                self.remove_hooks(hook_function)

    def _call_tile_hooks(self, tile: str) -> None:
        self.ground = tile
        for func in self._tile_event_hooks:
            func(self, tile)

    def _call_accelerometer_hooks(self, x: int, y: int, z: int) -> None:
        self.acceleration = (x, y, z)
        for func in self._accelerometer_hooks:
            func(self, x, y, z)

    def _call_pants_hooks(self, powerup: str) -> None:
        self.pants = powerup
        for func in self._pants_event_hooks:
            func(self, powerup)

    def _handle_events(self, sender: int, data: bytearray) -> None:
        """Handles bluetooth notifications.

        Decodes the sent data and calls Mario's appropriate event hooks.

        Args:
            sender (int): Only necessary for bleak compatibility
            data (bytearray): The data of the notification
        """
        hex_data = data.hex()
        # Port Value
        if data[2] == 0x45:
            # Camera Sensor Data
            if data[3] == 0x01:
                if data[4] == data[5] == data[6] == data[7] == 0xFF:
                    self.log(f"IDLE?, Hex: {hex_data}")
                elif data[4] == data[5] == 0xFF:
                    # Ground Colors
                    color = HEX_TO_COLOR_TILE.get(
                        data[6], f"Unkown Color: {hex(data[6])}"
                    )
                    self.log(f"{color} Ground, Hex: {hex_data}")
                    self._call_tile_hooks(color)
                else:
                    # RGB code
                    tile_code = int.from_bytes(data[4:], "big")
                    tile_name = HEX_TO_RGB_TILE.get(
                        tile_code, f"Unkown Tile Code: {hex(tile_code)}"
                    )
                    self.recent_tile = tile_name
                    self.log(f"{tile_name} Tile, Hex: {hex_data}")
                    self._call_tile_hooks(tile_name)

            # Accelerometer data
            elif data[3] == 0x00:
                # Gesture Mode - experimental, likely not accurate
                if data[4:6] == data[6:]:
                    gesture = ""
                    integer_data = int.from_bytes(data[4:6], "big")
                    for binary, name in BINARY_GESTURES.items():
                        if integer_data & binary:
                            gesture += name
                    self.log(gesture)

                # RAW Mode
                else:
                    try:
                        x = int(signed(data[4]))
                        y = int(signed(data[5]))
                        z = int(signed(data[6]))
                    except IndexError:
                        error_msg = (
                            f"Message length is {len(data)}, expected 6."
                            "The most likely cause is outdated software. To update, "
                            "connect your Lego Mario to the Lego Mario smartphone app."
                        )
                        self.log(error_msg)
                        raise ValueError(error_msg)
                    self.log(f"X: {x} Y: {y} Z: {z}", end="")
                    self._call_accelerometer_hooks(x, y, z)

            # Pants data
            elif data[3] == 0x02:
                pants = HEX_TO_PANTS.get(data[4], "Unkown")
                binary_pants = bin(data[4])
                self.log(
                    f"{pants} Pants, "
                    f"Pants-Only Binary: {binary_pants},"
                    f"Hex: {hex_data}"
                )
                self._call_pants_hooks(pants)
            # Port 3 data - uncertain about all of it
            elif data[3] == 0x03:
                if data[4] == 0x13 and data[5] == 0x01:
                    tile_name = HEX_TO_RGB_TILE.get(data[6], "Unkown Tile")
                    self.log(f"Port 3: Jumped on {tile_name}, Hex: {hex_data}")
                else:
                    # TBD
                    self.log(
                        f"Unknown value from port 3: {data[4:].hex()}, "
                        f"Hex: {hex_data}"
                    )
            else:
                self.log(
                    f"Unknown value from port {data[3]}: "
                    f"{data[4:].hex()}, Hex: {hex_data}"
                )

        # other technical messages
        elif data[2] == 0x02:  # Hub Actions
            action = HEX_TO_HUB_ACTIONS.get(
                data[3], f"Unkown Hub Action, Hex: {hex_data}"
            )
            self.log(f"{action}, Hex: {hex_data}")
            if data[3] == 0x31:  # 0x31 = Hub Will Disconnect
                asyncio.get_event_loop().create_task(self.disconnect())
        elif data[2] == 0x04:  # Hub Attached I/O
            if data[4]:
                self.log(f"Port {data[3]} got attached, Hex: {hex_data}")
            else:
                self.log(
                    f"Port {data[3]} got detached, "
                    f"this shouldn't happen. Hex: {hex_data}"
                )
        elif data[2] == 0x47:  # Port Input Format Handshake
            self.log(
                f"Port {data[3]} changed to mode {data[4]} "
                f"with{'out' if not data[9] else ''} notifications, "
                f"Hex: {hex_data}"
            )
        elif data[2] == 0x01 and data[4] == 0x06:
            hub_property = HEX_TO_HUB_PROPERTIES.get(data[3], "Unknown Property")
            self.log(
                f"Hub Update About {hub_property}: "
                f"{data[5:].hex()}, "
                f"Hex: {hex_data}"
            )
        else:  # Other
            self.log(
                f"Unknown message - check Lego Wireless Protocol, " f"Hex: {hex_data}"
            )

    async def connect(self) -> bool:
        self.run = True
        retries = 0
        while self.run:
            retries += 1
            if retries > 3:
                self.log("Stopped after 3 attempts, disconnecting...")
                break
            self.log("Searching for device...")
            try:
                devices = await BleakScanner.discover()
            except OSError as e:
                raise OSError(
                    "Can't use device - make sure your device"
                    f" supports Bluetooth and turn it on.\n{e}"
                )
            for d in devices:
                if d.name and (
                    d.name.lower().startswith("lego luigi")
                    or d.name.lower().startswith("lego mario")
                    or d.name.lower().startswith("lego peach")
                ):
                    try:
                        client = BleakClient(d)
                        await client.connect()
                        self.client = client
                        self.log(f"Mario Connected: {client.address}")

                        # subscribe to events
                        await client.start_notify(
                            LEGO_CHARACTERISTIC_UUID, self._handle_events
                        )
                        await asyncio.sleep(0.1)
                        await client.write_gatt_char(
                            LEGO_CHARACTERISTIC_UUID, SUBSCRIBE_IMU_COMMAND
                        )
                        await asyncio.sleep(0.1)
                        await client.write_gatt_char(
                            LEGO_CHARACTERISTIC_UUID, SUBSCRIBE_RGB_COMMAND
                        )
                        await asyncio.sleep(0.1)
                        await client.write_gatt_char(
                            LEGO_CHARACTERISTIC_UUID, SUBSCRIBE_PANTS_COMMAND
                        )

                        asyncio.get_event_loop().create_task(
                            self._check_connection_loop()
                        )

                        if not self.default_volume is None:
                            self.set_volume(self.default_volume)
                        return True
                    except Exception as ex:
                        if "-2140864507" in ex:
                            self.log(
                                "Mario's newest Firmware needs to be paired with your computer once before connecting. If you're on mac, please do so manually in settings. On Windows, this will now be done automatically. Try reconnecting"
                            )
                            await client.pair()
                        self.log(f"Error connecting: {ex}")
                        await self.disconnect()
                        return False
        await self.disconnect()
        return False

    async def request_port_value(self, port: int = 0) -> None:
        """Method for sending request for color sensor port value to Mario.
        Default port is 0.
        0 - Accelerometer
        1 - Camera
        2 - Pants
        3 - unknown
        4 - unknown
        6 - voltage?
        Response will be sent to event handlers.

        Args:
            port (int, optional): Port to request value from. Defaults to 0.
        """
        assert port in (0, 1, 2, 3, 4, 6), "Use a supported port (0,1,2,3,4,6)"
        if self.client:
            try:
                command = REQUEST_RGB_COMMAND
                command[3] = port
                command = bytearray(command)
                await self.client.write_gatt_char(LEGO_CHARACTERISTIC_UUID, command)
            except (OSError, BleakError):
                self.log("Connection error while requesting port value")
                await self.disconnect()

    def set_volume(self, new_volume: int) -> None:
        """Sets mario's volume to the specified volume.

        Args:
            new_volume (int): Percentage of maximum volume.
                Values <0 or >100 will be set to 0 or 100 respectively."""
        new_volume = min(max(new_volume, 0), 100)
        if self.client:
            try:
                command = bytearray([*MUTE_COMMAND[:5], new_volume])
                asyncio.get_event_loop().create_task(
                    self.client.write_gatt_char(LEGO_CHARACTERISTIC_UUID, command)
                )
            except (OSError, BleakError):
                self.log("Connection error while setting volume")
                asyncio.get_event_loop().create_task(self.disconnect())

    async def port_setup(
        self, port: int, mode: int, notifications: bool = True
    ) -> None:
        """Configures the settings of one of Mario's ports.
        Sends a message to Mario that configures the way one of its ports
        communicates.

        Args:
        port (int): The designated Port.
            Port 0: Accelerometer
            Port 1: Camera
            Port 2: Binary (Pants)
            Port 3: ??
            Port 4: ??
        mode (int): The mode to set the port to.
            Available modes:
                Port 0: (0,1),
                Port 1: (0,1),
                Port 2: (0),
                Port 3: (0,1,2,3),
                Port 4: (0,1).
            Also see https://github.com/bricklife/LEGO-Mario-Reveng
        notifications (bool, optional): Whether to receive updates about
            new values of the port. Defaults to True.
            If False, you'll need to manually request port values.
        """
        await self.await_connection()
        if self.client:
            try:
                command = pifs_command(port, mode, notifications)
                await self.client.write_gatt_char(LEGO_CHARACTERISTIC_UUID, command)
            except (OSError, BleakError):
                self.log("Connection error while setting up port")
                await self.disconnect()

    async def _check_connection_loop(self) -> None:
        while self.client:
            try:
                if not self.client.is_connected:
                    self.log("Disconnect detected during connection check")
                    await self.disconnect()
                await asyncio.sleep(3)
            except (OSError, BleakError):
                self.log("Error during connection check")
                await self.disconnect()

    async def disconnect(self) -> None:
        try:
            self.log("Disconnecting... ")
            if self.client:
                await self.client.write_gatt_char(
                    LEGO_CHARACTERISTIC_UUID, DISCONNECT_COMMAND
                )
                await self.client.disconnect()
                self.client = None
        except (OSError, BleakError):
            self.log("Connection error while disconnecting")
            self.client = None
        if self.auto_reconnect:
            asyncio.get_event_loop().create_task(self.connect())
        else:
            self.run = False

    async def turn_off(self) -> None:
        if not self.client:
            return
        try:
            self.log("Turning Off... ")
            await self.client.write_gatt_char(
                LEGO_CHARACTERISTIC_UUID, TURN_OFF_COMMAND
            )
            await self.disconnect()
        except (OSError, BleakError):
            self.log("Connection error while turning off")
            await self.disconnect()

    async def await_connection(self):
        while not self.is_connected:
            await asyncio.sleep(0.5)
        return

    @property
    def is_connected(self):
        if self.client:
            return self.client.is_connected
        return False

    def __str__(self) -> str:
        if self.is_connected:
            assert self.client
            return f"Mario at {self.client.address}"
        else:
            return "Mario - not connected"


def signed(char):
    return char - 256 if char > 127 else char


def run():
    """Runs the asyncio event loop until until all tasks are done."""
    while asyncio.all_tasks(loop=asyncio.get_event_loop()):
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*asyncio.all_tasks(loop=asyncio.get_event_loop()))
        )
