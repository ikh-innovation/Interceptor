import asyncio
import random
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw


async def run():

    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to PX4")
            break

    # Wait for good health
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Drone ready")
            break

    # ARM
    print("Arming...")
    await drone.action.arm()

    # START OFFBOARD MUST BE PRE-FED WITH SETPOINTS
    print("Starting offboard...")

    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -2.0, 0.0)
    )

    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        return

    print("OFFBOARD started")

    # RANDOM WALK LOOP
    while True:

        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        z = random.uniform(-3, -1.5)

        print(f"Moving to: {x:.2f}, {y:.2f}, {z:.2f}")

        await drone.offboard.set_position_ned(
            PositionNedYaw(x, y, z, 0.0)
        )

        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(run())