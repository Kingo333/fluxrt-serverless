"""Composite keep rules, reference cleaning and the seg cache mask (pure numpy, no GPU)."""
import numpy as np
import pytest

from livecam import pacing
from livecam import segment as sg

H, W = 64, 48


def scene(face=True, two_faces=False, feet=True, garment=(24, 40)):
    """Camera-frame segmentation: background, a face (or two), torso clothes, legs, feet (others)."""
    cat = np.zeros((H, W), dtype=np.uint8)
    if face:
        cat[4:12, 18:30] = sg.FACE
        cat[2:5, 16:32] = sg.HAIR
    if two_faces:
        cat[4:12, 2:10] = sg.FACE
    cat[12:16, 20:28] = sg.BODY_SKIN
    cat[garment[0]:garment[1], 14:34] = sg.CLOTHES
    cat[40:56, 18:30] = sg.BODY_SKIN   # legs
    if feet:
        cat[56:62, 16:32] = sg.OTHERS   # shoes
    return cat


def frames():
    real = np.full((H, W, 3), (10, 20, 200), dtype=np.uint8)
    gen = np.full((H, W, 3), (200, 20, 10), dtype=np.uint8)
    return real, gen


def test_normalize_category():
    assert sg.normalize_category("tops") == "garment" and sg.normalize_category("Dresses") == "garment"
    assert sg.normalize_category("upper") == "garment" and sg.normalize_category("outerwear") == "garment"
    assert sg.normalize_category("shoes") == "shoes" and sg.normalize_category("bags") == "bags"
    assert sg.normalize_category("accessories") == "accessories"
    assert sg.normalize_category(None) is None and sg.normalize_category("hat") is None


def test_shoes_without_face_is_pure_pass_through():
    cat = scene(face=False)
    real, gen = frames()
    keep = sg.keep_mask(cat, cat, "shoes")
    assert keep is None
    assert np.array_equal(sg.composite(gen, real, keep, 3), gen)


def test_shoes_with_face_keeps_face_and_generates_feet():
    cat = scene(face=True)
    real, gen = frames()
    keep = sg.keep_mask(cat, cat, "shoes")
    assert keep[8, 24] and keep[3, 20]            # face + hair kept
    assert not keep[58, 24] and not keep[0, 0]    # feet and background generated
    out = sg.composite(gen, real, keep, 0)
    assert tuple(out[8, 24]) == (10, 20, 200) and tuple(out[58, 24]) == (200, 20, 10)


def test_bag_without_face_passes_through_and_with_face_keeps_face_only():
    assert sg.keep_mask(scene(face=False), scene(face=False), "bags") is None
    keep = sg.keep_mask(scene(), scene(), "bags")
    assert keep[8, 24] and not keep[3, 20] and not keep[0, 0]


def test_accessory_with_face_is_pass_through():
    assert sg.keep_mask(scene(), scene(), "accessories") is None
    assert sg.keep_mask(scene(), scene(), None) is None          # unknown category too


def test_top_without_face_keeps_background_only_and_generates_torso():
    cat = scene(face=False)
    keep = sg.keep_mask(cat, cat, "tops")
    assert keep[0, 0] and not keep[30, 24] and not keep[13, 24]  # background yes, torso no, skin no


def test_top_generated_clothes_over_camera_background_win():
    cat_in = scene()
    cat_out = scene(garment=(16, 50))                              # the new garment is longer
    keep = sg.keep_mask(cat_in, cat_out, "tops")
    assert keep[0, 0] and keep[8, 24]                              # background and face kept
    assert not keep[45, 24] and not keep[18, 16]                   # extended garment: generated wins
    assert not keep[13, 24]                                        # body skin never kept


def test_two_faces_both_kept():
    cat = scene(two_faces=True)
    keep = sg.keep_mask(cat, cat, "dresses")
    assert keep[8, 24] and keep[8, 5]


def test_no_person_or_shape_mismatch_passes_through():
    empty = np.zeros((H, W), dtype=np.uint8)
    assert sg.keep_mask(empty, empty, "tops") is None
    assert sg.keep_mask(scene(), np.zeros((H, W + 1), dtype=np.uint8), "tops") is None


def test_composite_feather_blends_edge():
    cat = scene()
    real, gen = frames()
    keep = sg.keep_mask(cat, cat, "tops")
    out = sg.composite(gen, real, keep, 3)
    assert tuple(out[0, 0]) == (10, 20, 200)                      # deep background: real
    assert tuple(out[32, 24]) == (200, 20, 10)                     # deep garment: generated
    edge = out[24, 24].astype(int)                                 # garment top edge: a mix
    assert 10 < edge[0] < 200
    with pytest.raises(ValueError):
        sg.composite(gen, real[:-1], keep, 3)


def test_resize_category_is_nearest():
    cat = scene()
    big = sg.resize_category(cat, H * 2, W * 2)
    assert big.shape == (H * 2, W * 2) and set(np.unique(big)) == set(np.unique(cat))
    assert big[16, 48] == cat[8, 24]


def test_clean_reference_only_for_on_model_photos():
    rgb = np.full((H, W, 3), 90, dtype=np.uint8)
    flat = np.zeros((H, W), dtype=np.uint8)
    flat[10:50, 10:40] = sg.CLOTHES                                # ghost/flat-lay photo: no person
    assert sg.clean_reference(rgb, flat) is None
    on_model = scene()
    out = sg.clean_reference(rgb, on_model, feather_px=0)
    assert tuple(out[30, 24]) == (90, 90, 90) and tuple(out[8, 24]) == (255, 255, 255)
    tiny = scene(garment=(24, 26))                                 # garment < 5%: never blank it
    assert sg.clean_reference(rgb, tiny) is None


def test_cache_mask_shoes_is_full_and_top_skips_static_background():
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    b = sg.CacheMaskBuilder(H, W, cell=16, dilate=0, refresh=1000)
    m = b.build(frame, scene(), "shoes")
    assert m.shape == (4, 3) and m.dtype == np.int32 and (m == 2).all()
    b = sg.CacheMaskBuilder(H, W, cell=16, dilate=0, refresh=1000)
    assert (b.build(frame, scene(), "tops") == 2).all()           # first frame: everything
    m = b.build(frame, scene(), "tops")                            # static frame: only the person
    assert m[0, 0] == 0 and m[1, 1] == 2 and m[2, 1] == 2
    assert (b.build(frame, scene(), None) == 2).all()              # unknown category: full
    assert (b.build(frame, None, "tops") == 2).all()               # no segmentation: full
    b.force_full()
    assert (b.build(frame, scene(), "tops") == 2).all()


def test_cache_mask_motion_marks_moving_background():
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    b = sg.CacheMaskBuilder(H, W, cell=16, dilate=0, refresh=1000)
    b.build(frame, scene(), "tops")
    moved = frame.copy()
    moved[0:16, 0:16] = 250                                        # a bright change in a background cell
    m = b.build(moved, scene(), "tops")
    assert m[0, 0] == 2
    m2 = b.build(moved, scene(), "tops")                           # cached frame updated: now static
    assert m2[0, 0] == 0


def test_cache_mask_refresh_every_n_frames():
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    b = sg.CacheMaskBuilder(H, W, cell=16, dilate=0, refresh=3)
    fulls = [(b.build(frame, scene(), "tops") == 2).all() for _ in range(6)]
    assert fulls == [True, False, True, False, False, True]


def test_pacing_change_detector_and_ring():
    d = pacing.FrameChangeDetector(stride=8)
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    assert d.changed(a) and not d.changed(a)
    a[0, 0] = 1
    assert d.changed(a)
    r = pacing.FrameRing(size=4)
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        r.push(t, f"f{t}")
    assert len(r) == 4 and r.latest() == "f5.0" and r.at(2.9) == "f3.0" and r.at(0.0) == "f2.0"
    assert pacing.composite_source_ts(10.0, 0.25) == 9.75
    assert pacing.composite_source_ts(10.0, 0.25, lag_override_s=0.0) == 10.0
    assert pacing.FrameRing().at(1.0) is None


def test_load_segmenter_missing_model_returns_none(tmp_path):
    assert sg.load_segmenter(str(tmp_path / "missing.tflite")) is None
