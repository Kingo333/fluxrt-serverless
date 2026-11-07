"""Face-only paste (FLUXRT_FACE_PASTE) and the refreshing anchor (FLUXRT_ANCHOR_REFRESH_S).

Owner's checklist: skin tones and lighting, glasses, beards, hijab / headscarf / cap, two
people, and no-face shots (shoes at the feet, bags, watches) where nothing is pasted and the
item changes fully.
"""
import numpy as np

from livecam import segment as sg
from livecam import session as ss
from livecam.session import LiveSession

H, W = 64, 48


def blank():
    return np.zeros((H, W), dtype=np.uint8)


def person(cat, top=10, left=16, h=16, w=14, cap=False, scalp=True, beard=False, glasses=False, hijab=False):
    """Draw one person: face box, optional cap / scalp hair / beard / glasses / hijab, body below."""
    if hijab:
        cat[max(0, top - 4):top + h + 4, max(0, left - 4):left + w + 4] = sg.CLOTHES
    elif scalp:
        cat[max(0, top - 3):top + 2, left:left + w] = sg.HAIR
    if cap:
        cat[max(0, top - 5):top + 1, left - 1:left + w + 1] = sg.OTHERS
    cat[top:top + h, left:left + w] = sg.FACE
    if beard:
        cat[top + h - 5:top + h + 2, left + 2:left + w - 2] = sg.HAIR
    if glasses:
        cat[top + 4:top + 7, left + 2:left + w - 2] = sg.OTHERS
    cat[top + h + 2:min(H, top + h + 30), max(0, left - 8):left + w + 8] = sg.CLOTHES
    return cat


def test_aligned_face_pastes_face_only_never_cap_hair_or_background():
    cam = person(blank(), cap=True)
    gen = person(blank(), cap=True)
    paste, fade = sg.face_paste_mask(cam, gen, "tops", 0.6)
    assert paste is not None and fade is not None
    assert paste[18, 22]                                  # middle of the face
    assert not paste[cam == sg.OTHERS].any()              # cap
    assert not paste[cam == sg.HAIR].any()                # scalp hair
    assert not paste[cam == sg.BACKGROUND].any()
    assert not paste[cam == sg.CLOTHES].any()


def test_glasses_and_beard_count_as_face_scalp_hair_does_not():
    cam = person(blank(), beard=True, glasses=True)
    paste, _ = sg.face_paste_mask(cam, cam.copy(), "tops", 0.6)
    assert paste[15, 22]                                  # glasses lens row: filled in, real eyes kept
    assert paste[27, 22]                                  # beard below the face skin
    assert not paste[8, 22]                               # scalp hair above the face


def test_hijab_and_generated_scarf_are_never_covered():
    cam = person(blank(), hijab=True)
    gen = person(blank(), hijab=True)
    gen[22:26, 16:30] = sg.CLOTHES                        # FluxRT drew the scarf over the chin
    paste, _ = sg.face_paste_mask(cam, gen, "tops", 0.5)
    assert paste is not None and paste[14, 22]
    assert not paste[22:26].any()                         # the generated scarf stays
    assert not paste[cam == sg.CLOTHES].any()             # the real scarf is not pasted either


def test_two_people_each_pasted_only_when_they_line_up():
    cam = person(person(blank(), left=4, w=12), left=30, w=12)
    gen = person(person(blank(), left=4, w=12), left=30, w=12)
    paste, _ = sg.face_paste_mask(cam, gen, "tops", 0.6)
    assert paste[18, 9] and paste[18, 35]
    moved = person(person(blank(), left=4, w=12), left=36, w=12, top=20)   # second person moved
    paste, fade = sg.face_paste_mask(cam, moved, "tops", 0.6)
    assert paste[18, 9] and not paste[18, 35]             # only the face that lines up
    assert fade[18, 9]


def test_misaligned_face_is_not_pasted_only_faded_where_both_see_a_face():
    cam = person(blank(), top=10)
    gen = person(blank(), top=18)                         # head moved fast: half overlap
    paste, fade = sg.face_paste_mask(cam, gen, "tops", 0.6)
    assert paste is None
    assert fade is not None and not fade[cam == sg.BACKGROUND].any() and not fade[11, 22]
    far = person(blank(), top=40)
    assert sg.face_paste_mask(cam, far, "tops", 0.6) == (None, None)


def test_no_face_shots_and_face_items_never_paste():
    feet = blank()
    feet[40:64, 10:38] = sg.BODY_SKIN                      # shoes shot: feet only
    assert sg.face_paste_mask(feet, feet.copy(), "shoes", 0.6) == (None, None)
    hand = blank()
    hand[20:40, 10:30] = sg.BODY_SKIN                      # watch / bag: hand or arm only
    assert sg.face_paste_mask(hand, hand.copy(), "bags", 0.6) == (None, None)
    face = person(blank())
    for category in ("accessories", None, "", "hats"):     # may sit on the face: FluxRT keeps it all
        assert sg.face_paste_mask(face, face.copy(), category, 0.6) == (None, None)
    for category in ("tops", "dresses", "shoes", "bags"):   # face fine to keep for these
        assert sg.face_paste_mask(face, face.copy(), category, 0.6)[0] is not None


def test_tiny_far_away_face_is_left_to_fluxrt():
    cat = blank()
    cat[5:8, 20:23] = sg.FACE                              # 9 px of 3072
    assert sg.face_regions(cat) == []


def test_alpha_ends_inside_the_outline_and_never_changes_skin_colour():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    a = sg.face_alpha(mask, shrink_px=2, feather_px=4)
    assert a[20, 20] == 1.0 and a[5, 20] == 0.0 and a[6, 20] == 0.0     # edge rows stay generated
    assert 0.0 < a[10, 20] < 1.0                                         # soft edge
    for tone in ((40, 30, 25), (120, 90, 70), (225, 200, 185)):          # deep to light skin, any light
        real = np.full((40, 40, 3), tone, dtype=np.uint8)
        gen = np.full((40, 40, 3), (90, 150, 210), dtype=np.uint8)
        out = sg.blend(gen, real, a)
        assert tuple(out[20, 20]) == tone                                # the real face, untouched


# ---- LiveSession -----------------------------------------------------------------

def crop(frame, h, w):
    ys = (np.arange(h) * frame.shape[0] / h).astype(int)
    xs = (np.arange(w) * frame.shape[1] / w).astype(int)
    return np.ascontiguousarray(frame[ys[:, None], xs[None, :]])


class FakeProc:
    def __init__(self):
        self.anchors = []
        self.set_anchor_image = lambda img: self.anchors.append(None if img is None else img.shape)

    def set_mask(self, m):
        pass

    def get_last_processing_time(self):
        return 0.0


class SceneSeg:
    """Camera frames are blue-ish, FluxRT frames red-ish; each gets its own category map."""
    def __init__(self, cam_cat, gen_cat):
        self.cam_cat, self.gen_cat = cam_cat, gen_cat

    def __call__(self, rgb):
        cat = self.gen_cat if rgb[..., 0].mean() > rgb[..., 2].mean() else self.cam_cat
        return sg.resize_category(cat, rgb.shape[0], rgb.shape[1])


CAM = (200, 20, 10)     # BGR: blue camera
GEN = (10, 20, 200)     # BGR: red FluxRT output


def make(cam_cat, gen_cat, **kw):
    kw.setdefault("face_feather", 2)
    kw.setdefault("face_shrink", 1)
    s = LiveSession(FakeProc(), crop, SceneSeg(cam_cat, gen_cat), width=W, height=H,
                    out_width=W * 2, out_height=H * 2, face_paste_on=True, **kw)
    s.set_category("tops")
    return s


def run(s, t):
    s.prepare_input(np.full((H * 2, W * 2, 3), CAM, dtype=np.uint8), now=t)
    return s.finish_output(np.full((H * 2, W * 2, 3), GEN, dtype=np.uint8), now=t + 0.05, last_processing_time=0.05)


def test_session_fades_the_real_face_in_and_leaves_everything_else_generated():
    cat = person(blank(), cap=True)
    s = make(cat, cat.copy())
    out = run(s, 1.0)
    assert 0 < s.face_gain < 1 and tuple(out[0, 0]) == GEN
    for i in range(ss.FACE_FADE_IN):
        out = run(s, 1.1 + i * 0.05)
    assert s.face_gain == 1.0
    assert tuple(out[36, 44]) == CAM                         # face centre: the real face
    assert tuple(out[10, 44]) == GEN                         # cap: FluxRT's
    assert tuple(out[80, 44]) == GEN                         # new outfit: FluxRT's
    assert tuple(out[0, 0]) == GEN                           # background: FluxRT's
    assert s.stats["face_pasted"] >= 1


def test_session_fades_out_when_the_faces_stop_lining_up_and_stops_with_no_face():
    cat = person(blank())
    s = make(cat, cat.copy())
    for i in range(ss.FACE_FADE_IN):
        run(s, 1.0 + i * 0.05)
    s.segmenter.gen_cat = person(blank(), top=17)           # fast head turn
    run(s, 2.0)
    assert 0 < s.face_gain < 1
    for i in range(ss.FACE_FADE_OUT):
        run(s, 2.1 + i * 0.05)
    assert s.face_gain == 0.0
    s.segmenter.cam_cat = s.segmenter.gen_cat = blank()     # walked away
    out = run(s, 3.0)
    assert s.face_gain == 0.0 and tuple(out[36, 44]) == GEN


def test_session_no_face_item_changes_fully():
    feet = blank()
    feet[40:64, 10:38] = sg.BODY_SKIN
    s = make(feet, feet.copy())
    s.set_category("shoes")
    gen = None
    for i in range(8):
        gen = run(s, 1.0 + i * 0.05)
    assert (gen == np.array(GEN, dtype=np.uint8)).all() and s.stats["face_pasted"] == 0


def test_face_paste_off_by_default_and_yields_to_the_composite():
    cat = person(blank())
    s = LiveSession(FakeProc(), crop, SceneSeg(cat, cat), width=W, height=H, out_width=W * 2, out_height=H * 2)
    assert not s.face_paste_on and "face_pasted" not in s.stats
    s = LiveSession(FakeProc(), crop, SceneSeg(cat, cat), width=W, height=H, out_width=W * 2, out_height=H * 2,
                    composite_on=True, face_paste_on=True)
    assert s.composite_on and not s.face_paste_on
    s = LiveSession(FakeProc(), crop, None, width=W, height=H, out_width=W, out_height=H, face_paste_on=True)
    assert not s.face_paste_on                               # no segmenter: off


def test_anchor_refreshes_every_n_seconds_but_never_on_an_empty_frame():
    cat = person(blank())
    s = LiveSession(FakeProc(), crop, SceneSeg(cat, cat), width=W, height=H, out_width=W * 2, out_height=H * 2,
                    anchor_on=True, anchor_delay_s=3.0, anchor_refresh_s=4.0)
    gen = np.full((H * 2, W * 2, 3), GEN, dtype=np.uint8)
    s.garment_changed(now=0.0)
    s.finish_output(gen, now=3.0)
    assert len(s.proc.anchors) == 1
    s.finish_output(gen, now=6.9)
    assert len(s.proc.anchors) == 1
    s.finish_output(gen, now=7.0)
    assert len(s.proc.anchors) == 2                          # refreshed after 4 s
    s.segmenter.gen_cat = blank()                            # nobody in the frame
    s.finish_output(gen, now=11.0)
    assert len(s.proc.anchors) == 2                          # kept the last good one
    s.segmenter.gen_cat = cat
    s.finish_output(gen, now=11.5)
    assert len(s.proc.anchors) == 2                          # looks again a second later
    s.finish_output(gen, now=12.0)
    assert len(s.proc.anchors) == 3


def test_anchor_refresh_zero_keeps_one_anchor_per_garment():
    s = LiveSession(FakeProc(), crop, None, width=W, height=H, out_width=W, out_height=H,
                    anchor_on=True, anchor_delay_s=1.0)
    gen = np.zeros((H, W, 3), dtype=np.uint8)
    s.garment_changed(now=0.0)
    for t in (1.0, 5.0, 60.0):
        s.finish_output(gen, now=t)
    assert len(s.proc.anchors) == 1


def test_paste_face_works_in_the_face_box_only_and_matches_the_full_frame_blend():
    rng = np.random.default_rng(0)
    gen = rng.integers(0, 255, (128, 96, 3), dtype=np.uint8)
    real = rng.integers(0, 255, (128, 96, 3), dtype=np.uint8)
    mask = np.zeros((128, 96), dtype=bool)
    mask[30:80, 20:70] = True
    out = sg.paste_face(gen, real, mask, 4, 8, 0.7)
    full = sg.blend(gen, real, sg.face_alpha(mask, 4, 8) * 0.7)
    assert np.array_equal(out, full) and out is not gen
    tiny = np.zeros((128, 96), dtype=bool)
    tiny[40:45, 40:45] = True                                  # gone after shrinking
    assert sg.paste_face(gen, real, tiny, 4, 8) is None
    assert sg.paste_face(gen, real, mask, 4, 8, 0.0) is None
