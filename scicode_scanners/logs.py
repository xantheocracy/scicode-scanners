"""Compatibility for Inspect logs using Zstandard ZIP compression."""

import zipfile


def enable_zstd_zip():
    """Support Inspect's Zstandard ZIP members on older Python versions."""
    if getattr(zipfile, '_scicode_zstd_enabled', False):
        return
    import zstandard

    original = zipfile._get_decompressor
    original_check = zipfile._check_compression

    def decompress(compression):
        if compression in (20, 93):
            return zstandard.ZstdDecompressor().decompressobj()
        return original(compression)

    def check(compression):
        if compression not in (20, 93):
            original_check(compression)

    zipfile._get_decompressor = decompress
    zipfile._check_compression = check
    zipfile._scicode_zstd_enabled = True
