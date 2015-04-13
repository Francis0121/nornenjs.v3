S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 16711
Date: 2015-04-13 09:39:37+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xb71b70bc

Register Information
r0   = 0xaf178200, r1   = 0xb71b209c
r2   = 0x00000800, r3   = 0xb71b70bc
r4   = 0x00000010, r5   = 0xb71b209c
r6   = 0xaf178000, r7   = 0x00000080
r8   = 0x00000800, r9   = 0xb71b209c
r10  = 0x00000200, fp   = 0x00000040
ip   = 0x00000000, sp   = 0xb0dfda88
lr   = 0xb3b2d924, pc   = 0xb3b2daec
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    357840 KB
Buffers:     72200 KB
Cached:     149136 KB
VmPeak:     143448 KB
VmSize:     119024 KB
VmLck:           0 KB
VmHWM:       15844 KB
VmRSS:       15844 KB
VmData:      63048 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       25068 KB
VmPTE:          82 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 16711 TID = 16720
16711 16715 16716 16717 16718 16719 16720 16721 

Maps Information
b1794000 b1795000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17d2000 b17d3000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17db000 b17e2000 r-xp /usr/lib/libfeedback.so.0.1.4
b17f5000 b17f6000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b17fe000 b1815000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b199b000 b19a0000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31e9000 b3234000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b323d000 b3247000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3250000 b32ac000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3ab9000 b3abf000 r-xp /usr/lib/libUMP.so
b3ac7000 b3ace000 r-xp /usr/lib/libtbm.so.1.0.0
b3ad6000 b3adb000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3ae3000 b3aea000 r-xp /usr/lib/libdrm.so.2.4.0
b3af3000 b3af5000 r-xp /usr/lib/libdri2.so.0.0.0
b3afd000 b3b10000 r-xp /usr/lib/libEGL_platform.so
b3b19000 b3bf0000 r-xp /usr/lib/libMali.so
b3bfb000 b3c02000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3c0a000 b3c0f000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3c17000 b3c2e000 r-xp /usr/lib/libEGL.so.1.4
b3c37000 b3c3c000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c45000 b3c46000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c4f000 b3c67000 r-xp /usr/lib/libpng12.so.0.50.0
b3c6f000 b3cad000 r-xp /usr/lib/libGLESv2.so.2.0
b3cb5000 b3cb9000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3cc2000 b3cc5000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3cce000 b3d84000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d8f000 b3da5000 r-xp /usr/lib/libtts.so
b3dae000 b3dbf000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3dc7000 b3dc9000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3dd1000 b3dd9000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3de1000 b3eb0000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3ebb000 b3ec5000 r-xp /lib/libnss_files-2.13.so
b3ed3000 b3ed5000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b40de000 b40ff000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b4108000 b4125000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b412e000 b41fc000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4213000 b4239000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4243000 b4245000 r-xp /usr/lib/libiniparser.so.0
b424f000 b4255000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b425e000 b4264000 r-xp /usr/lib/libappsvc.so.0.1.0
b426d000 b426f000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4278000 b427c000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b4284000 b4288000 r-xp /usr/lib/libogg.so.0.7.1
b4290000 b42b2000 r-xp /usr/lib/libvorbis.so.0.4.3
b42ba000 b439e000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b43b2000 b43e3000 r-xp /usr/lib/libFLAC.so.8.2.0
b43ec000 b43ee000 r-xp /usr/lib/libXau.so.6.0.0
b43f6000 b4442000 r-xp /usr/lib/libssl.so.1.0.0
b444f000 b447d000 r-xp /usr/lib/libidn.so.11.5.44
b4485000 b448f000 r-xp /usr/lib/libcares.so.2.1.0
b4497000 b44dc000 r-xp /usr/lib/libsndfile.so.1.0.25
b44ea000 b44f1000 r-xp /usr/lib/libsensord-share.so
b44f9000 b450f000 r-xp /lib/libexpat.so.1.5.2
b451d000 b4520000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b4528000 b455c000 r-xp /usr/lib/libicule.so.51.1
b4565000 b4578000 r-xp /usr/lib/libxcb.so.1.1.0
b4580000 b45bb000 r-xp /usr/lib/libcurl.so.4.3.0
b45c4000 b45cd000 r-xp /usr/lib/libethumb.so.1.7.99
b5b3b000 b5bcf000 r-xp /usr/lib/libstdc++.so.6.0.16
b5be2000 b5be4000 r-xp /usr/lib/libctxdata.so.0.0.0
b5bec000 b5bf9000 r-xp /usr/lib/libremix.so.0.0.0
b5c01000 b5c02000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c0a000 b5c21000 r-xp /usr/lib/liblua-5.1.so
b5c2a000 b5c31000 r-xp /usr/lib/libembryo.so.1.7.99
b5c39000 b5c5c000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c74000 b5c8a000 r-xp /usr/lib/libsensor.so.1.1.0
b5c93000 b5ce9000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5cf6000 b5d19000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d22000 b5d68000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d71000 b5d84000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d8c000 b5ddc000 r-xp /usr/lib/libfreetype.so.6.8.1
b5de7000 b5dea000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5df2000 b5df6000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5dfe000 b5e03000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e0c000 b5e16000 r-xp /usr/lib/libXext.so.6.4.0
b5e1e000 b5eff000 r-xp /usr/lib/libX11.so.6.3.0
b5f0a000 b5f0d000 r-xp /usr/lib/libXtst.so.6.1.0
b5f15000 b5f1b000 r-xp /usr/lib/libXrender.so.1.3.0
b5f23000 b5f28000 r-xp /usr/lib/libXrandr.so.2.2.0
b5f30000 b5f31000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f3a000 b5f42000 r-xp /usr/lib/libXi.so.6.1.0
b5f43000 b5f46000 r-xp /usr/lib/libXfixes.so.3.1.0
b5f4e000 b5f50000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f58000 b5f5a000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f62000 b5f63000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f6c000 b5f72000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f7b000 b5f94000 r-xp /usr/lib/libecore_con.so.1.7.99
b5f9e000 b5fa4000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5fac000 b5fb4000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5fbc000 b5fc0000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5fc9000 b5fdf000 r-xp /usr/lib/libefreet.so.1.7.99
b5fe8000 b5ff1000 r-xp /usr/lib/libedbus.so.1.7.99
b5ff9000 b60de000 r-xp /usr/lib/libicuuc.so.51.1
b60f3000 b6232000 r-xp /usr/lib/libicui18n.so.51.1
b6242000 b629e000 r-xp /usr/lib/libedje.so.1.7.99
b62a8000 b62b9000 r-xp /usr/lib/libecore_input.so.1.7.99
b62c1000 b62c6000 r-xp /usr/lib/libecore_file.so.1.7.99
b62ce000 b62e7000 r-xp /usr/lib/libeet.so.1.7.99
b62f8000 b62fc000 r-xp /usr/lib/libappcore-common.so.1.1
b6305000 b63d1000 r-xp /usr/lib/libevas.so.1.7.99
b63f6000 b6417000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6420000 b644f000 r-xp /usr/lib/libecore_x.so.1.7.99
b6459000 b658d000 r-xp /usr/lib/libelementary.so.1.7.99
b65a5000 b65a6000 r-xp /usr/lib/libjournal.so.0.1.0
b65af000 b667a000 r-xp /usr/lib/libxml2.so.2.7.8
b6688000 b6698000 r-xp /lib/libresolv-2.13.so
b669c000 b66b2000 r-xp /lib/libz.so.1.2.5
b66ba000 b66bc000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b66c4000 b66c9000 r-xp /usr/lib/libffi.so.5.0.10
b66d2000 b66d3000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b66db000 b66de000 r-xp /lib/libattr.so.1.1.0
b66e6000 b688e000 r-xp /usr/lib/libcrypto.so.1.0.0
b68ae000 b68c8000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b68d1000 b693a000 r-xp /lib/libm-2.13.so
b6943000 b6983000 r-xp /usr/lib/libeina.so.1.7.99
b698c000 b6994000 r-xp /usr/lib/libvconf.so.0.2.45
b699c000 b699f000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b69a7000 b69db000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b69e4000 b6ab8000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6ac4000 b6aca000 r-xp /lib/librt-2.13.so
b6ad3000 b6ad8000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6ae1000 b6ae8000 r-xp /lib/libcrypt-2.13.so
b6b18000 b6b1b000 r-xp /lib/libcap.so.2.21
b6b23000 b6b25000 r-xp /usr/lib/libiri.so
b6b2d000 b6b4c000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b54000 b6b6a000 r-xp /usr/lib/libecore.so.1.7.99
b6b80000 b6b85000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b8e000 b6c5e000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c5f000 b6c6d000 r-xp /usr/lib/libail.so.0.1.0
b6c75000 b6c8c000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c95000 b6c9f000 r-xp /lib/libunwind.so.8.0.1
b6ccd000 b6de8000 r-xp /lib/libc-2.13.so
b6df6000 b6dfe000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e06000 b6e30000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e39000 b6e3c000 r-xp /usr/lib/libbundle.so.0.1.22
b6e44000 b6e46000 r-xp /lib/libdl-2.13.so
b6e4f000 b6e52000 r-xp /usr/lib/libsmack.so.1.0.0
b6e5a000 b6ebc000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6ec6000 b6ed8000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6ee1000 b6ef5000 r-xp /lib/libpthread-2.13.so
b6f02000 b6f06000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f10000 b6f12000 r-xp /usr/lib/libdlog.so.0.0.0
b6f1a000 b6f25000 r-xp /usr/lib/libaul.so.0.1.0
b6f2f000 b6f33000 r-xp /usr/lib/libsys-assert.so
b6f3c000 b6f59000 r-xp /lib/ld-2.13.so
b6f62000 b6f68000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f70000 b6f9c000 rw-p [heap]
b6f9c000 b71b7000 rw-p [heap]
bec46000 bec67000 rwxp [stack]
End of Maps Information

Callstack Information (PID:16711)
Call Stack Count: 1
 0: (0xb3b2daec) [/usr/lib/libMali.so] + 0x14aec
End of Call Stack

Package Information
Package Name: org.tizen.other
Package ID : org.tizen.other
Version: 1.0.0
Package Type: coretpk
App Name: other
App ID: org.tizen.other
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
g.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 60
04-13 09:39:27.585+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 09:39:27.585+0900 D/PKGMGR  (16648): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126
04-13 09:39:27.585+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-13 09:39:27.585+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(419) > event_cb is called
04-13 09:39:27.585+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:27.585+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:27.585+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 60
04-13 09:39:27.585+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 09:39:27.585+0900 D/QUICKPANEL( 2279): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:60
04-13 09:39:27.585+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:27.585+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:27.590+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 60
04-13 09:39:27.590+0900 D/PKGMGR  ( 2276): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 09:39:27.590+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 60
04-13 09:39:27.590+0900 D/DATA_PROVIDER_MASTER( 2276): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 60
04-13 09:39:27.590+0900 D/PKGMGR  ( 2198): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 09:39:27.590+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:27.590+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:27.590+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:27.590+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:27.590+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 60
04-13 09:39:27.590+0900 D/PKGMGR  ( 2274): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 09:39:27.590+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:27.590+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:27.590+0900 E/PKGMGR_CERT(16650): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 2 14
04-13 09:39:27.590+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 60
04-13 09:39:27.590+0900 D/PKGMGR  ( 2245): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 09:39:27.590+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:60] for org.tizen.other
04-13 09:39:27.590+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 60
04-13 09:39:27.590+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:27.590+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:27.590+0900 E/PKGMGR_CERT(16650): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 10 3
04-13 09:39:27.590+0900 E/PKGMGR_CERT(16650): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 4 5
04-13 09:39:27.590+0900 E/PKGMGR_CERT(16650): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 5 5
04-13 09:39:27.590+0900 E/PKGMGR_CERT(16650): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 6 5
04-13 09:39:27.605+0900 E/PKGMGR_CERT(16650): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(571) > Transaction Commit and End
04-13 09:39:27.605+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/res]
04-13 09:39:27.605+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_make_directory(1490) > mkdir failed. appdir=[/usr/apps/org.tizen.other/shared], errno=[2][No such file or directory]
04-13 09:39:27.605+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared]
04-13 09:39:27.605+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/opt/usr/apps/org.tizen.other/shared/data]
04-13 09:39:27.605+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared/res]
04-13 09:39:27.610+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/tizen-manifest.xml]
04-13 09:39:27.610+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/author-signature.xml]
04-13 09:39:27.610+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/signature1.xml]
04-13 09:39:27.610+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/share/packages/org.tizen.other.xml]
04-13 09:39:27.610+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_make_directory_for_ext(1353) > Directory dose not exist. path: /opt/usr/apps/org.tizen.other/shared/data, errno: 2 (No such file or directory)
04-13 09:39:27.705+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_register_package(64) > [smack] app_install(org.tizen.other), result=[0]
04-13 09:39:27.710+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other, 5, _), result=[0]
04-13 09:39:27.710+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared, 5, _), result=[0]
04-13 09:39:27.710+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/res, 5, _), result=[0]
04-13 09:39:27.710+0900 E/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_get_smack_label_access(627) > Error in getting smack ACCESS label failed. result[-1] (path:[/opt/usr/apps/org.tizen.other/shared/data]))
04-13 09:39:27.720+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_get_group_id(1770) > encoding done, len=[28]
04-13 09:39:27.720+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_apply_smack(1878) > groupid = [QVI9+j1mlN94KHCZf1TZSIDuLJk=] for shared/trusted.
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/trusted, 1, QVI9+j1mlN94KHCZf1TZSIDuLJk=), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/bin, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/lib, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/tizen-manifest.xml, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/author-signature.xml, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/signature1.xml, 0, org.tizen.other), result=[0]
04-13 09:39:27.820+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/share/packages/org.tizen.other.xml, 0, org.tizen.other), result=[0]
04-13 09:39:27.825+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other, 5, _), result=[0]
04-13 09:39:27.825+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-13 09:39:27.825+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-13 09:39:27.825+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/shared, 5, _), result=[0]
04-13 09:39:27.830+0900 D/rpm-installer(16650): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/internet]
04-13 09:39:27.855+0900 D/rpm-installer(16650): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-13 09:39:27.860+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-13 09:39:27.960+0900 D/rpm-installer(16650): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-13 09:39:27.960+0900 D/rpm-installer(16650): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-13 09:39:27.960+0900 D/rpm-installer(16650): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/systemsettings]
04-13 09:39:27.980+0900 D/rpm-installer(16650): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-13 09:39:27.985+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-13 09:39:28.085+0900 D/rpm-installer(16650): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-13 09:39:28.085+0900 D/rpm-installer(16650): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-13 09:39:28.100+0900 D/rpm-installer(16650): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-13 09:39:28.100+0900 D/rpm-installer(16650): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-13 09:39:28.180+0900 D/rpm-installer(16650): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-13 09:39:28.180+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_package_install(2194) > permission applying done successfully.
04-13 09:39:28.180+0900 D/PRIVILEGE_INFO(16650): privilege_info.c: privilege_manager_verify_privilege_list(625) > privilege_info_compare_privilege_level called
04-13 09:39:28.180+0900 D/PRIVILEGE_INFO(16650): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/internet
04-13 09:39:28.180+0900 D/PRIVILEGE_INFO(16650): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/systemsettings
04-13 09:39:28.180+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_verify_privilege_list(672) > privilege_manager_verify_privilege_list(PRVMGR_PACKAGE_TYPE_CORE) is ok.
04-13 09:39:28.180+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_package_install(2202) > _coretpk_installer_verify_privilege_list done.
04-13 09:39:28.185+0900 D/rpm-installer(16650): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[100]
04-13 09:39:28.185+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_package_install(2224) > install status is [2].
04-13 09:39:28.185+0900 D/rpm-installer(16650): coretpk-installer.c: __post_install_for_mmc(311) > Installed storage is internal.
04-13 09:39:28.185+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_package_install(2231) > _coretpk_installer_package_install is done.
04-13 09:39:28.185+0900 D/rpm-installer(16650): rpm-vconf-intf.c: _ri_broadcast_status_notification(196) > pkgid=[org.tizen.other], key=[end], val=[ok]
04-13 09:39:28.185+0900 D/rpm-installer(16650): coretpk-installer.c: _coretpk_installer_prepare_package_install(2695) > success
04-13 09:39:28.185+0900 D/rpm-installer(16650): rpm-appcore-intf.c: main(224) > sync() start
04-13 09:39:28.185+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.185+0900 D/PKGMGR  ( 2400): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.185+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.185+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.185+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.185+0900 D/PKGMGR  ( 2245): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.185+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:100] for org.tizen.other
04-13 09:39:28.185+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 100
04-13 09:39:28.185+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.185+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.185+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.185+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.185+0900 D/PKGMGR  (16648): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126
04-13 09:39:28.185+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-13 09:39:28.185+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(419) > event_cb is called
04-13 09:39:28.185+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.185+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.190+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.190+0900 D/PKGMGR  ( 2276): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.190+0900 D/DATA_PROVIDER_MASTER( 2276): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 100
04-13 09:39:28.190+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.190+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.190+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.190+0900 D/PKGMGR  ( 2274): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.190+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.190+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.190+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / install_percent / 100
04-13 09:39:28.190+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 09:39:28.190+0900 D/QUICKPANEL( 2279): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:100
04-13 09:39:28.190+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.190+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.190+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.190+0900 D/PKGMGR  ( 2170): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [install] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.190+0900 D/PKGMGR  ( 2170): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.190+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(538) > [SECURE_LOG] pkgid(org.tizen.other), key(end), value(ok)
04-13 09:39:28.190+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(559) > [SECURE_LOG] op(install), value(ok)
04-13 09:39:28.195+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2276): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.195+0900 D/DATA_PROVIDER_MASTER( 2276): pkgmgr.c: end_cb(409) > [SECURE_LOG] [org.tizen.other] ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2245): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.195+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [end:ok] for org.tizen.other
04-13 09:39:28.195+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _end(520) > Package(org.tizen.other) : key(install) - val(ok)
04-13 09:39:28.195+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2274): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.195+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.195+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.195+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.195+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.195+0900 D/PKGMGR  (16648): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126
04-13 09:39:28.195+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-13 09:39:28.195+0900 D/PKGMGR  (16648): pkgmgr.c: __operation_callback(419) > event_cb is called
04-13 09:39:28.195+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.195+0900 D/PKGMGR  (16648): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.195+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2400): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.195+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.195+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.195+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126 / coretpk / org.tizen.other / end / ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1486028126] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 09:39:28.195+0900 D/QUICKPANEL( 2279): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:end val:ok
04-13 09:39:28.195+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.195+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.215+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.215+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.230+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __app_info_insert_handler(185) > __app_info_insert_handler
04-13 09:39:28.230+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __app_info_insert_handler(388) > [SECURE_LOG] appinfo file:org.tizen.other, comp:ui, type:rpm
04-13 09:39:28.230+0900 D/PKGMGR  ( 2170): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.230+0900 D/PKGMGR  ( 2170): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:28.255+0900 D/MENU_SCREEN( 2245): layout.c: layout_create_package(200) > package org.tizen.other is installed directly
04-13 09:39:28.565+0900 D/MENU_SCREEN( 2245): item.c: item_update(594) > Access to file [/opt/usr/apps/org.tizen.other/shared/res/other.png], size[57662]
04-13 09:39:28.595+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
04-13 09:39:28.600+0900 D/rpm-installer(16650): rpm-appcore-intf.c: main(226) > sync() end
04-13 09:39:28.605+0900 D/rpm-installer(16650): rpm-appcore-intf.c: main(245) > ------------------------------------------------
04-13 09:39:28.605+0900 D/rpm-installer(16650): rpm-appcore-intf.c: main(246) >  [END] rpm-installer: result=[0]
04-13 09:39:28.605+0900 D/rpm-installer(16650): rpm-appcore-intf.c: main(247) > ------------------------------------------------
04-13 09:39:28.640+0900 D/PKGMGR_SERVER(16640): pkgmgr-server.c: sighandler(326) > child exit [16650]
04-13 09:39:28.640+0900 D/PKGMGR_SERVER(16640): pkgmgr-server.c: sighandler(341) > child NORMAL exit [16650]
04-13 09:39:28.690+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 09:39:28.690+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 09:39:29.185+0900 D/PKGMGR_SERVER(16640): pkgmgr-server.c: exit_server(724) > exit_server Start
04-13 09:39:29.185+0900 D/PKGMGR_SERVER(16640): pkgmgr-server.c: main(1516) > Quit main loop.
04-13 09:39:29.185+0900 D/PKGMGR_SERVER(16640): pkgmgr-server.c: main(1524) > package manager server terminated.
04-13 09:39:35.605+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 0
04-13 09:39:35.605+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-13 09:39:35.610+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 16708, pid = 16710
04-13 09:39:35.620+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-13 09:39:35.625+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
04-13 09:39:35.625+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-13 09:39:35.625+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-13 09:39:35.625+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-13 09:39:35.625+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-13 09:39:35.625+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-13 09:39:35.625+0900 D/AUL_PAD ( 2204): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-13 09:39:35.630+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 16711 /opt/usr/apps/org.tizen.other/bin/other
04-13 09:39:35.630+0900 D/AUL_PAD (16711): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-13 09:39:35.630+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-13 09:39:35.630+0900 D/AUL_PAD (16711): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-13 09:39:35.630+0900 D/AUL_PAD (16711): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-13 09:39:35.650+0900 D/AUL_PAD (16711): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-13 09:39:35.650+0900 D/AUL_PAD (16711): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-13 09:39:35.650+0900 D/AUL_PAD (16711): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-13 09:39:35.650+0900 D/AUL_PAD (16711): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-13 09:39:35.650+0900 D/LAUNCH  (16711): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-13 09:39:35.680+0900 I/CAPI_APPFW_APPLICATION(16711): app_main.c: ui_app_main(697) > app_efl_main
04-13 09:39:35.680+0900 D/LAUNCH  (16711): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-13 09:39:35.725+0900 D/APP_CORE(16711): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-13 09:39:35.730+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-13 09:39:35.730+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-13 09:39:35.730+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-13 09:39:35.730+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-13 09:39:35.730+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-13 09:39:35.730+0900 D/RESOURCED( 2372): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-13 09:39:35.730+0900 D/RESOURCED( 2372): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
04-13 09:39:35.730+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 16711, type 4 
04-13 09:39:35.730+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 16711
04-13 09:39:35.730+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-13 09:39:35.730+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 16711
04-13 09:39:35.730+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-13 09:39:35.730+0900 D/RESOURCED( 2372): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 16711
04-13 09:39:35.730+0900 D/RESOURCED( 2372): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 16711
04-13 09:39:35.730+0900 D/RESOURCED( 2372): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-13 09:39:35.730+0900 D/RESOURCED( 2372): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-13 09:39:35.730+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 09:39:35.730+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 09:39:35.730+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 09:39:35.730+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 09:39:35.730+0900 E/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 519
04-13 09:39:35.730+0900 D/RESOURCED( 2372): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-13 09:39:35.740+0900 D/AUL     (16711): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 16711 is org.tizen.other
04-13 09:39:35.740+0900 D/APP_CORE(16711): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-13 09:39:35.740+0900 D/APP_CORE(16711): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-13 09:39:35.740+0900 D/AUL     (16711): app_sock.c: __create_server_sock(135) > pg path - already exists
04-13 09:39:35.740+0900 D/LAUNCH  (16711): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-13 09:39:35.740+0900 I/CAPI_APPFW_APPLICATION(16711): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-13 09:39:35.990+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=16711
04-13 09:39:36.025+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-13 09:39:36.030+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
04-13 09:39:36.030+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-13 09:39:36.030+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
04-13 09:39:36.030+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-13 09:39:36.035+0900 F/socket.io(16711): thread_start
04-13 09:39:36.035+0900 F/socket.io(16711): finish 0
04-13 09:39:36.035+0900 D/LAUNCH  (16711): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-13 09:39:36.035+0900 D/APP_CORE(16711): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-13 09:39:36.035+0900 D/APP_CORE(16711): appcore.c: __aul_handler(423) > [APP 16711]     AUL event: AUL_START
04-13 09:39:36.035+0900 D/APP_CORE(16711): appcore-efl.c: __do_app(470) > [APP 16711] Event: RESET State: CREATED
04-13 09:39:36.035+0900 D/APP_CORE(16711): appcore-efl.c: __do_app(496) > [APP 16711] RESET
04-13 09:39:36.035+0900 D/LAUNCH  (16711): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-13 09:39:36.035+0900 I/CAPI_APPFW_APPLICATION(16711): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-13 09:39:36.035+0900 D/APP_SVC (16711): appsvc.c: __set_bundle(161) > __set_bundle
04-13 09:39:36.035+0900 D/LAUNCH  (16711): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-13 09:39:36.040+0900 F/socket.io(16711): Connect Start
04-13 09:39:36.040+0900 F/socket.io(16711): Set ConnectListener
04-13 09:39:36.040+0900 F/socket.io(16711): Set ClosetListener
04-13 09:39:36.040+0900 F/socket.io(16711): Set FaileListener
04-13 09:39:36.040+0900 F/socket.io(16711): Connect
04-13 09:39:36.040+0900 F/socket.io(16711): Lock
04-13 09:39:36.040+0900 F/socket.io(16711): !!!
04-13 09:39:36.050+0900 I/APP_CORE(16711): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-13 09:39:36.050+0900 I/APP_CORE(16711): appcore-efl.c: __do_app(509) > [APP 16711] Initial Launching, call the resume_cb
04-13 09:39:36.050+0900 I/CAPI_APPFW_APPLICATION(16711): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-13 09:39:36.050+0900 D/APP_CORE(16711): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-13 09:39:36.060+0900 D/APP_CORE(16711): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600003
04-13 09:39:36.060+0900 D/APP_CORE(16711): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600003
04-13 09:39:36.185+0900 F/other   (16711): success!!!! -1223401960 -1223401960 
04-13 09:39:36.200+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-13 09:39:36.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.upload"
04-13 09:39:36.240+0900 D/APP_CORE(16711): appcore.c: __prt_ltime(183) > [APP 16711] first idle after reset: 637 msec
04-13 09:39:36.245+0900 E/socket.io(16711): 566: Connected.
04-13 09:39:36.250+0900 D/sio_packet(16711): from json
04-13 09:39:36.250+0900 D/value.IsObject()(16711): if binary from json
04-13 09:39:36.250+0900 D/sio_packet(16711): from json
04-13 09:39:36.250+0900 D/sio_packet(16711): from json
04-13 09:39:36.250+0900 D/value.IsArray()(16711): if arr from json
04-13 09:39:36.250+0900 D/sio_packet(16711): from json
04-13 09:39:36.250+0900 D/sio_packet(16711): IsInt64
04-13 09:39:36.250+0900 D/sio_packet(16711): from json
04-13 09:39:36.250+0900 D/sio_packet(16711): IsInt64
04-13 09:39:36.250+0900 E/socket.io(16711): 554: On handshake, sid
04-13 09:39:36.250+0900 E/socket.io(16711): 651: Received Message type(connect)
04-13 09:39:36.250+0900 E/socket.io(16711): 489: On Connected
04-13 09:39:36.250+0900 F/socket.io(16711): unlock
04-13 09:39:36.250+0900 F/socket.io(16711): emit connectMessage
04-13 09:39:36.250+0900 F/sio_packet(16711): accept()
04-13 09:39:36.250+0900 F/sio_packet(16711): start accept_message
04-13 09:39:36.250+0900 F/sio_packet(16711): arr, but we will use it like binary
04-13 09:39:36.250+0900 F/sio_packet(16711): start accept_array_message()
04-13 09:39:36.250+0900 F/sio_packet(16711): start accept_message
04-13 09:39:36.250+0900 F/sio_packet(16711): start accept_message
04-13 09:39:36.255+0900 E/socket.io(16711): 743: encoded paylod length: 63
04-13 09:39:36.255+0900 F/socket.io(16711): bind connectMessage
04-13 09:39:36.255+0900 F/socket.io(16711): emit connectMessage
04-13 09:39:36.255+0900 F/sio_packet(16711): accept()
04-13 09:39:36.255+0900 F/sio_packet(16711): start accept_message
04-13 09:39:36.255+0900 F/sio_packet(16711): arr, but we will use it like binary
04-13 09:39:36.255+0900 F/sio_packet(16711): start accept_array_message()
04-13 09:39:36.255+0900 F/sio_packet(16711): start accept_message
04-13 09:39:36.255+0900 F/sio_packet(16711): start accept_message
04-13 09:39:36.255+0900 E/socket.io(16711): 743: encoded paylod length: 13
04-13 09:39:36.255+0900 F/socket.io(16711): close 16711
04-13 09:39:36.255+0900 E/socket.io(16711): 800: ping exit, con is expired? 0, ec: Operation canceled
04-13 09:39:36.255+0900 E/socket.io(16711): 800: ping exit, con is expired? 0, ec: Operation canceled
04-13 09:39:36.255+0900 D/APP_CORE( 2245): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-13 09:39:36.255+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-13 09:39:36.255+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-13 09:39:36.255+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(470) > [APP 2245] Event: PAUSE State: RUNNING
04-13 09:39:36.255+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(538) > [APP 2245] PAUSE
04-13 09:39:36.255+0900 I/CAPI_APPFW_APPLICATION( 2245): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-13 09:39:36.255+0900 D/MENU_SCREEN( 2245): menu_screen.c: _pause_cb(538) > Pause start
04-13 09:39:36.255+0900 D/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 09:39:36.255+0900 E/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 09:39:36.260+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2245, type = 2
04-13 09:39:36.260+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 16711, type = 0
04-13 09:39:36.260+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(16711) status(3)
04-13 09:39:36.260+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 16711
04-13 09:39:36.260+0900 I/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 16711, oom : 200
04-13 09:39:36.260+0900 E/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-13 09:39:36.290+0900 D/sio_packet(16711): from json
04-13 09:39:36.290+0900 D/value.IsArray()(16711): if arr from json
04-13 09:39:36.290+0900 D/sio_packet(16711): from json
04-13 09:39:36.290+0900 D/sio_packet(16711): from json
04-13 09:39:36.290+0900 D/value.IsObject()(16711): if binary from json
04-13 09:39:36.290+0900 D/sio_packet(16711): from json
04-13 09:39:36.290+0900 D/sio_packet(16711): from json
04-13 09:39:36.290+0900 D/sio_packet(16711): from json
04-13 09:39:36.290+0900 E/socket.io(16711): 669: Received Message type(Event)
04-13 09:39:36.290+0900 F/socket.io(16711): bind_event [connectMessage] 16711
04-13 09:39:36.290+0900 F/socket.io(16711): connectMessage :: 
04-13 09:39:36.290+0900 D/APP_CORE(16711): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 0
04-13 09:39:36.290+0900 D/APP_CORE(16711): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-13 09:39:36.290+0900 D/APP_CORE(16711): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-13 09:39:36.295+0900 D/APP_CORE(16711): appcore-efl.c: __do_app(470) > [APP 16711] Event: RESUME State: RUNNING
04-13 09:39:36.295+0900 D/LAUNCH  (16711): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-13 09:39:36.295+0900 D/LAUNCH  (16711): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-13 09:39:36.295+0900 D/LAUNCH  (16711): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-13 09:39:36.295+0900 D/APP_CORE(16711): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 09:39:36.295+0900 E/APP_CORE(16711): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 09:39:36.740+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.other /opt/usr/apps/org.tizen.other/bin/other
04-13 09:39:36.740+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
04-13 09:39:36.750+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
04-13 09:39:37.195+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-13 09:39:37.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
04-13 09:39:37.430+0900 D/sio_packet(16711): from json
04-13 09:39:37.430+0900 D/value.IsArray()(16711): if arr from json
04-13 09:39:37.430+0900 D/sio_packet(16711): from json
04-13 09:39:37.430+0900 D/sio_packet(16711): from json
04-13 09:39:37.430+0900 D/value.IsObject()(16711): if binary from json
04-13 09:39:37.430+0900 D/sio_packet(16711): from json
04-13 09:39:37.430+0900 D/value.IsObject()(16711): if binary from json
04-13 09:39:37.430+0900 D/sio_packet(16711): from json
04-13 09:39:37.430+0900 D/sio_packet(16711): IsInt64
04-13 09:39:37.430+0900 D/sio_packet(16711): from json
04-13 09:39:37.430+0900 D/value.IsObject()(16711): if binary from json
04-13 09:39:37.430+0900 F/value.IsObject()(16711): num!!!!!0
04-13 09:39:37.430+0900 F/value.IsObject()(16711): buf size!!!!!1
04-13 09:39:37.430+0900 E/socket.io(16711): 669: Received Message type(Event)
04-13 09:39:37.430+0900 F/socket.io(16711): bind_event [test2] 16711
04-13 09:39:37.430+0900 F/socket.io(16711): key size : 9312
04-13 09:39:37.430+0900 F/get_binary(16711): in get binary_message()...
04-13 09:39:37.430+0900 F/socket.io(16711): texture_getter value -1276401648
04-13 09:39:37.430+0900 F/other   (16711): another success 255 216 255 224 0 38 2 40
04-13 09:39:37.670+0900 W/CRASH_MANAGER(16722): worker.c: worker_job(1189) > 11167116f7468142888557
