import asyncio
import random
import math

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

    # ============================================================
    # WAIT FOR POSITION ESTIMATE
    # ============================================================

    async for health in drone.telemetry.health():

        if (
            health.is_global_position_ok
            and health.is_home_position_ok
        ):
            print("Drone ready")
            break

    # ============================================================
    # ARM
    # ============================================================

    print("Arming...")

    await drone.action.arm()

    # ============================================================
    # INITIAL OFFBOARD SETPOINT
    # ============================================================

    await drone.offboard.set_position_ned(
        PositionNedYaw(
            0.0,
            0.0,
            -5.0,
            0.0
        )
    )

    print("Starting offboard...")

    try:

        await drone.offboard.start()

    except OffboardError as e:

        print(
            f"Offboard failed: "
            f"{e._result.result}"
        )

        return

    print("OFFBOARD started")

    # ============================================================
    # OLD RANDOM WALK (KEPT AS COMMENT)
    # ============================================================

    """
    while True:

        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        z = random.uniform(-3, -1.5)

        await drone.offboard.set_position_ned(
            PositionNedYaw(x, y, z, 0.0)
        )

        await asyncio.sleep(2)
    """

    # ============================================================
    # FAST RANDOM CIRCLES (KEPT AS COMMENT)
    # ============================================================

    """
    while True:

        radius = random.uniform(2.0, 5.0)

        center_x = random.uniform(-5.0, 5.0)
        center_y = random.uniform(-5.0, 5.0)

        z = random.uniform(-3.0, -1.5)

        angular_speed = random.uniform(4.0, 8.0)

        for step in range(60):

            theta = step * 0.2 * angular_speed

            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)

            await drone.offboard.set_position_ned(
                PositionNedYaw(x, y, z, 0.0)
            )

            await asyncio.sleep(0.02)
    """

    # ============================================================
    # FAST BACK/FORTH ROUTE (KEPT AS COMMENT)
    # ============================================================

    """
    while True:

        x1 = random.uniform(-15.0, -5.0)
        x2 = random.uniform(5.0, 15.0)

        y = random.uniform(-5.0, 5.0)

        z = random.uniform(-3.0, -1.5)

        for t in range(50):

            alpha = t / 49.0

            x = x1 + (x2 - x1) * alpha

            await drone.offboard.set_position_ned(
                PositionNedYaw(x, y, z, 0.0)
            )

            await asyncio.sleep(0.01)

        for t in range(50):

            alpha = t / 49.0

            x = x2 + (x1 - x2) * alpha

            await drone.offboard.set_position_ned(
                PositionNedYaw(x, y, z, 180.0)
            )

            await asyncio.sleep(0.01)
    """

    # ============================================================
    # FAST RANDOM LONG-DISTANCE ROUTES (KEPT AS COMMENT)
    # ============================================================

    """
    current_x = 0.0
    current_y = 0.0
    current_z = -2.0

    while True:

        target_x = random.uniform(-40.0, 40.0)
        target_y = random.uniform(-40.0, 40.0)
        target_z = random.uniform(-6.0, -2.0)

        dx = target_x - current_x
        dy = target_y - current_y
        dz = target_z - current_z

        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        steps = max(80, int(distance * 4))

        for step in range(steps):

            alpha = step / steps

            x = current_x + dx * alpha
            y = current_y + dy * alpha
            z = current_z + dz * alpha

            yaw = math.degrees(math.atan2(dy, dx))

            await drone.offboard.set_position_ned(
                PositionNedYaw(x, y, z, yaw)
            )

            await asyncio.sleep(0.01)

        current_x = target_x
        current_y = target_y
        current_z = target_z
    """

    # ============================================================
    # ACTIVE MODE:
    # ULTRA FAST RANDOM 3D ROUTES
    # ============================================================

    current_x = 0.0
    current_y = 0.0
    current_z = -5.0

    while True:

        # ========================================================
        # HUGE RANDOM TARGETS
        # ========================================================

        target_x = random.uniform(-120.0, 120.0)

        target_y = random.uniform(-150.0, 150.0)

        target_z = random.uniform(-25.0, -3.0)

        print(
            f"\nNEW RANDOM TARGET -> "
            f"x:{target_x:.1f} "
            f"y:{target_y:.1f} "
            f"z:{target_z:.1f}"
        )

        # ========================================================
        # DIRECTION VECTOR
        # ========================================================

        dx = target_x - current_x
        dy = target_y - current_y
        dz = target_z - current_z

        # Face movement direction
        yaw = math.degrees(
            math.atan2(dy, dx)
        )

        # ========================================================
        # AGGRESSIVE FAST MOTION
        # ========================================================

        steps = random.randint(15, 30)

        for step in range(steps):

            alpha = step / (steps - 1)

            # Base interpolation
            x = current_x + dx * alpha
            y = current_y + dy * alpha
            z = current_z + dz * alpha

            # Add random chaos/jitter
            x += random.uniform(-3.0, 3.0)
            y += random.uniform(-3.0, 3.0)
            z += random.uniform(-1.5, 1.5)

            await drone.offboard.set_position_ned(
                PositionNedYaw(
                    x,
                    y,
                    z,
                    yaw
                )
            )

            print(
                f"FAST RANDOM MOVE -> "
                f"x:{x:.2f} "
                f"y:{y:.2f} "
                f"z:{z:.2f}"
            )

            # Fast update rate
            await asyncio.sleep(0.01)

        # ========================================================
        # UPDATE CURRENT POSITION
        # ========================================================

        current_x = target_x
        current_y = target_y
        current_z = target_z


if __name__ == "__main__":

    asyncio.run(run())