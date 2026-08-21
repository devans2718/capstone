# Makes `tests` a package so test modules can import shared fixtures from each
# other (test_frame_geometry_model imports the recorded geometry from
# test_frame_geometry). Deleting this file breaks that import — it is not an
# accidental leftover.
