stage_a = Analysis(
    ['stage.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
stage_pyz = PYZ(stage_a.pure)

stage_exe = EXE(
    stage_pyz,
    stage_a.scripts,
    [],
    exclude_binaries=True,
    name='stage',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

summary_a = Analysis(
    ['summary.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
summary_pyz = PYZ(summary_a.pure)

summary_exe = EXE(
    summary_pyz,
    summary_a.scripts,
    [],
    exclude_binaries=True,
    name='summary',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

dpd_a = Analysis(
    ['analysis_dpd.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
dpd_pyz = PYZ(dpd_a.pure)

dpd_exe = EXE(
    dpd_pyz,
    dpd_a.scripts,
    [],
    exclude_binaries=True,
    name='analysis_dpd',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

slide_a = Analysis(
    ['slide.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
slide_pyz = PYZ(slide_a.pure)

slide_exe = EXE(
    slide_pyz,
    slide_a.scripts,
    [],
    exclude_binaries=True,
    name='slide',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

plot_hypno_a = Analysis(
    ['plot_hypnogram.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
plot_hypno_pyz = PYZ(plot_hypno_a.pure)

plot_hypno_exe = EXE(
    plot_hypno_pyz,
    plot_hypno_a.scripts,
    [],
    exclude_binaries=True,
    name='plot_hypnogram',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

plot_spec_a = Analysis(
    ['plot_spectrum.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
plot_spec_pyz = PYZ(plot_spec_a.pure)

plot_spec_exe = EXE(
    plot_spec_pyz,
    plot_spec_a.scripts,
    [],
    exclude_binaries=True,
    name='plot_spectrum',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

plot_ts_a = Analysis(
    ['plot_timeseries.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
plot_ts_pyz = PYZ(plot_ts_a.pure)

plot_ts_exe = EXE(
    plot_ts_pyz,
    plot_ts_a.scripts,
    [],
    exclude_binaries=True,
    name='plot_timeseries',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    stage_exe,
    stage_a.binaries,
    stage_a.datas,
    summary_exe,
    summary_a.binaries,
    summary_a.datas,
    dpd_exe,
    dpd_a.binaries,
    dpd_a.datas,
    slide_exe,
    slide_a.binaries,
    slide_a.datas,
    plot_hypno_exe,
    plot_hypno_a.binaries,
    plot_hypno_a.datas,
    plot_spec_exe,
    plot_spec_a.binaries,
    plot_spec_a.datas,
    plot_ts_exe,
    plot_ts_a.binaries,
    plot_ts_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='faster2_exe',
)