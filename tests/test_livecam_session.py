"""LiveSession with fakes: off = untouched frames; composite, cache mask and anchor behaviours."""
import numpy as np

from livecam import segment as sg
from livecam.session import LiveSession

W, H = 48, 64


def crop(frame, h, w):
    """Nearest-neighbour resize standing in for crop_maximal_rectangle (same aspect in tests)."""
    ys = (np.arange(h) * frame.shape[0] / h).astype(int)
    xs = (np.arange(w) * frame.shape[1] / w).astype(int)
    return np.ascontiguousarray(frame[ys[:, None], xs[None, :]])


class FakeProc:
    def __init__(self, with_anchor=True, lpt=0.2):
        self.masks, self.anchors, self.lpt = [], [], lpt
        if with_anchor:
            self.set_anchor_image = lambda img: self.anchors.append(None if img is None else img.shape)

    def set_mask(self, m):
        self.masks.append(m.copy())

    def get_last_processing_time(self):
        return self.lpt


def scene_cat(face=True):
    cat = np.zeros((H, W), dtype=np.uint8)
    if face:
        cat[4:12, 18:30] = sg.FACE
    cat[16:40, 14:34] = sg.CLOTHES
    cat[40:56, 18:30] = sg.BODY_SKIN
    return cat


def seg_factory(face=True):
    def seg(rgb):
        return sg.resize_category(scene_cat(face), rgb.shape[0], rgb.shape[1])
    return seg


def camera(color):
    return np.full((H * 2, W * 2, 3), color, dtype=np.uint8)


def test_all_switches_off_is_the_plain_crop():
    proc = FakeProc()
    s = LiveSession(proc, crop, None, width=W, height=H, out_width=W * 2, out_height=H * 2)
    frame = camera((10, 20, 200))
    resized = s.prepare_input(frame, now=1.0)
    assert resized.shape == (H, W, 3) and proc.masks == []
    out = np.full((H * 2, W * 2, 3), 7, dtype=np.uint8)
    assert s.finish_output(out, now=1.3) is out
    assert s.stats == {"composited": 0, "passed_through": 0, "masks": 0, "anchors": 0}


def test_composite_uses_the_camera_frame_that_produced_the_output():
    proc = FakeProc(lpt=0.2)
    s = LiveSession(proc, crop, seg_factory(), width=W, height=H, out_width=W * 2, out_height=H * 2,
                    composite_on=True, feather=0)
    s.set_category("tops")
    s.prepare_input(camera((10, 20, 200)), now=1.0)   # blue camera frame at t=1.0
    s.prepare_input(camera((0, 255, 0)), now=1.4)     # green frame at t=1.4
    gen = np.full((H * 2, W * 2, 3), (200, 20, 10), dtype=np.uint8)
    out = s.finish_output(gen, now=1.6)               # produced ~0.2 s ago -> the t=1.4 frame
    assert tuple(out[0, 0]) == (0, 255, 0)            # background: real (green)
    assert tuple(out[60, 48]) == (200, 20, 10)        # garment: generated
    assert tuple(out[16, 48]) == (0, 255, 0)          # face: real
    assert tuple(out[90, 48]) == (200, 20, 10)        # body skin: generated
    assert s.stats["composited"] == 1


def test_composite_shoes_without_face_passes_through_unchanged():
    proc = FakeProc()
    s = LiveSession(proc, crop, seg_factory(face=False), width=W, height=H, out_width=W * 2, out_height=H * 2,
                    composite_on=True)
    s.set_category("shoes")
    s.prepare_input(camera((10, 20, 200)), now=1.0)
    gen = np.full((H * 2, W * 2, 3), (200, 20, 10), dtype=np.uint8)
    out = s.finish_output(gen, now=1.2)
    assert out is gen and s.stats["passed_through"] == 1


def test_composite_without_any_input_yet_passes_through():
    s = LiveSession(FakeProc(), crop, seg_factory(), width=W, height=H, out_width=W * 2, out_height=H * 2,
                    composite_on=True)
    s.set_category("tops")
    gen = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
    assert s.finish_output(gen, now=1.0) is gen


def test_cache_mask_full_first_then_person_only_and_forced_on_garment_change():
    proc = FakeProc()
    s = LiveSession(proc, crop, seg_factory(), width=W, height=H, out_width=W, out_height=H,
                    cache_mask="seg", mask_dilate=0, mask_refresh=1000)
    s.set_category("tops")
    frame = camera((128, 128, 128))
    s.prepare_input(frame, now=1.0)
    s.prepare_input(frame, now=1.1)
    assert (proc.masks[0] == 2).all() and proc.masks[0].shape == (H // 16, W // 16)
    assert proc.masks[1][0, 0] == 0 and proc.masks[1][1, 1] == 2
    s.garment_changed(now=1.2)
    s.prepare_input(frame, now=1.3)
    assert (proc.masks[2] == 2).all()
    s.set_category("shoes")
    s.prepare_input(frame, now=1.4)
    assert (proc.masks[3] == 2).all()


def test_cache_mask_not_used_without_segmenter_or_when_auto():
    proc = FakeProc()
    s = LiveSession(proc, crop, None, width=W, height=H, out_width=W, out_height=H, cache_mask="seg")
    s.prepare_input(camera((1, 1, 1)), now=1.0)
    assert proc.masks == [] and s.mask_builder is None
    s2 = LiveSession(proc, crop, seg_factory(), width=W, height=H, out_width=W, out_height=H, cache_mask="auto")
    s2.prepare_input(camera((1, 1, 1)), now=1.0)
    assert proc.masks == []


def test_anchor_taken_after_delay_and_cleared_on_garment_change():
    proc = FakeProc()
    s = LiveSession(proc, crop, None, width=W, height=H, out_width=W * 2, out_height=H * 2,
                    anchor_on=True, anchor_delay_s=1.5)
    gen = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
    s.finish_output(gen, now=5.0)
    assert proc.anchors == []                          # no garment yet: nothing
    s.garment_changed(now=10.0)
    s.finish_output(gen, now=11.0)
    assert proc.anchors == []                          # before the delay
    s.finish_output(gen, now=11.6)
    assert proc.anchors == [(H, W, 3)] and s.anchor_set  # model-size RGB anchor
    s.finish_output(gen, now=12.0)
    assert len(proc.anchors) == 1                      # only once
    s.garment_changed(now=13.0)
    assert proc.anchors[-1] is None and not s.anchor_set


def test_anchor_off_when_fluxrt_has_no_patch():
    s = LiveSession(FakeProc(with_anchor=False), crop, None, width=W, height=H, out_width=W, out_height=H,
                    anchor_on=True)
    assert s.anchor_on is False


def test_category_normalisation_and_optional_calls_never_raise():
    class Broken(FakeProc):
        def set_mask(self, m):
            raise RuntimeError("queue closed")
    s = LiveSession(Broken(), crop, seg_factory(), width=W, height=H, out_width=W, out_height=H, cache_mask="seg")
    s.set_category("  Tops ")
    assert s.category == "Tops"
    s.set_category("")
    assert s.category is None
    s.prepare_input(camera((1, 1, 1)), now=1.0)        # set_mask raises inside: logged, not raised
