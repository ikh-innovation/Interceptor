class Renderer:
    def open(self, size):
        pass

    def draw(self, ctx, tracker, fused, roi, cmd, sent):
        pass

    def present(self, frame):
        return False

    def close(self):
        pass


class NullRenderer(Renderer):
    pass


class RendererFactory:
    @staticmethod
    def create(config, commander, skip_frames):
        if not config.enabled:
            return NullRenderer()
        from .hud import HudRenderer
        return HudRenderer(config, commander, skip_frames)
