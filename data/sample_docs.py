"""
Sample technical documentation fragments.

These sentences simulate the kind of content found in large-scale
audio/video documentation — the target domain for semantic deduplication.
Pairs of near-duplicates are intentionally seeded to validate the detector.
"""

SAMPLE_FRAGMENTS = [
    # --- Audio processing ---
    "Dolby Atmos renders audio objects in three-dimensional space using height channels.",
    "In Dolby Atmos, sound objects are positioned and rendered in a 3D listening environment that includes overhead speakers.",
    "The Atmos renderer places audio objects spatially, including the vertical dimension, for an immersive experience.",

    # --- Dynamic range ---
    "Dynamic range compression reduces the difference between the loudest and quietest parts of an audio signal.",
    "A dynamic range compressor lowers the volume of loud sounds and raises quiet ones to achieve a more uniform output level.",
    "Compression of dynamic range narrows the gap between peak and floor signal levels in audio content.",

    # --- Codec / encoding ---
    "The AC-4 codec delivers efficient audio compression with improved quality at lower bitrates.",
    "AC-4 is an audio coding standard that provides high-quality sound while significantly reducing the required bitrate.",
    "Dolby AC-4 achieves better audio fidelity than legacy codecs at equivalent or lower bitrates.",

    # --- HDR video ---
    "Dolby Vision uses per-scene or per-frame metadata to guide tone mapping on compatible displays.",
    "Per-frame dynamic metadata in Dolby Vision instructs the display on how to perform tone mapping for each scene.",
    "Tone mapping in Dolby Vision is driven by dynamic metadata attached to individual frames or scenes.",

    # --- Loudness ---
    "Content should be normalized to a target integrated loudness of -23 LUFS for broadcast delivery.",
    "Broadcast specifications require integrated loudness normalization to -23 LUFS before delivery.",
    "Before broadcast submission, audio must be normalized so that its integrated loudness measures -23 LUFS.",

    # --- Noise floor ---
    "The noise floor of a recording system defines the lowest signal level that can be captured above the background noise.",
    "Background noise inherent to a recording system sets its noise floor, the minimum usable signal threshold.",

    # --- Clearly distinct sentences (should NOT cluster) ---
    "HDMI 2.1 supports a maximum bandwidth of 48 Gbps for high-resolution video transmission.",
    "The Dolby TrueHD format supports up to 16 channels of lossless audio at 24-bit/192 kHz.",
    "Metadata embedded in the bitstream allows receivers to reconstruct the original mixing environment.",
    "A dialogue enhancement algorithm selectively boosts speech frequencies to improve intelligibility.",
    "Spatial audio rendering latency must remain below 10 milliseconds to avoid perceptible lip-sync errors.",
    "The decoder applies an inverse MDCT to reconstruct time-domain audio from spectral coefficients.",
]
