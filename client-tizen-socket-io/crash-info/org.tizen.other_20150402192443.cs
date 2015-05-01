S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 4096
Date: 2015-04-02 19:24:43+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xae0bc008

Register Information
r0   = 0xae4bf000, r1   = 0xae0bc008
r2   = 0x00000800, r3   = 0xae0bc008
r4   = 0x00000010, r5   = 0xae0bc008
r6   = 0xae4bf000, r7   = 0x00000000
r8   = 0x00000800, r9   = 0xae0bc008
r10  = 0x00000200, fp   = 0x00000000
ip   = 0x00000000, sp   = 0xbeef77f8
lr   = 0xb3b66910, pc   = 0xb3b66ae4
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    440348 KB
Buffers:     43204 KB
Cached:     122480 KB
VmPeak:     147912 KB
VmSize:     119900 KB
VmLck:           0 KB
VmHWM:       19104 KB
VmRSS:       16700 KB
VmData:      64076 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24920 KB
VmPTE:          86 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 4096 TID = 4096
4096 4100 4101 4102 4103 4104 4105 4106 

Maps Information
b1803000 b1804000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b1841000 b1842000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b184a000 b1851000 r-xp /usr/lib/libfeedback.so.0.1.4
b1864000 b1865000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b186d000 b1884000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b1a0a000 b1a0f000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b3258000 b32a3000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b32ac000 b32b6000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b32bf000 b331b000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3b28000 b3b2e000 r-xp /usr/lib/libUMP.so
b3b36000 b3b49000 r-xp /usr/lib/libEGL_platform.so
b3b52000 b3c29000 r-xp /usr/lib/libMali.so
b3c34000 b3c4b000 r-xp /usr/lib/libEGL.so.1.4
b3c54000 b3c59000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c62000 b3c63000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c6c000 b3c84000 r-xp /usr/lib/libpng12.so.0.50.0
b3c8c000 b3cca000 r-xp /usr/lib/libGLESv2.so.2.0
b3cd2000 b3cd6000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3cdf000 b3ce1000 r-xp /usr/lib/libdri2.so.0.0.0
b3ce9000 b3cf0000 r-xp /usr/lib/libdrm.so.2.4.0
b3cf9000 b3daf000 r-xp /usr/lib/libcairo.so.2.11200.14
b3dba000 b3dd0000 r-xp /usr/lib/libtts.so
b3dd9000 b3de0000 r-xp /usr/lib/libtbm.so.1.0.0
b3de8000 b3ded000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3df5000 b3e06000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3e0e000 b3e15000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3e1d000 b3e22000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3e2a000 b3e2c000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3e34000 b3e3c000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3e44000 b3e47000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3e50000 b3efa000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3f04000 b3f0e000 r-xp /lib/libnss_files-2.13.so
b3f1c000 b3f1e000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4127000 b4148000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b4151000 b416e000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b4177000 b4245000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b425c000 b4282000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b428c000 b428e000 r-xp /usr/lib/libiniparser.so.0
b4298000 b429e000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b42a7000 b42ad000 r-xp /usr/lib/libappsvc.so.0.1.0
b42b6000 b42b8000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b42c1000 b42c5000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b42cd000 b42d1000 r-xp /usr/lib/libogg.so.0.7.1
b42d9000 b42fb000 r-xp /usr/lib/libvorbis.so.0.4.3
b4303000 b43e7000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b43fb000 b442c000 r-xp /usr/lib/libFLAC.so.8.2.0
b4435000 b4437000 r-xp /usr/lib/libXau.so.6.0.0
b443f000 b448b000 r-xp /usr/lib/libssl.so.1.0.0
b4498000 b44c6000 r-xp /usr/lib/libidn.so.11.5.44
b44ce000 b44d8000 r-xp /usr/lib/libcares.so.2.1.0
b44e0000 b4525000 r-xp /usr/lib/libsndfile.so.1.0.25
b4533000 b453a000 r-xp /usr/lib/libsensord-share.so
b4542000 b4558000 r-xp /lib/libexpat.so.1.5.2
b4566000 b4569000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b4571000 b45a5000 r-xp /usr/lib/libicule.so.51.1
b45ae000 b45c1000 r-xp /usr/lib/libxcb.so.1.1.0
b45c9000 b4604000 r-xp /usr/lib/libcurl.so.4.3.0
b460d000 b4616000 r-xp /usr/lib/libethumb.so.1.7.99
b5b84000 b5c18000 r-xp /usr/lib/libstdc++.so.6.0.16
b5c2b000 b5c2d000 r-xp /usr/lib/libctxdata.so.0.0.0
b5c35000 b5c42000 r-xp /usr/lib/libremix.so.0.0.0
b5c4a000 b5c4b000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c53000 b5c6a000 r-xp /usr/lib/liblua-5.1.so
b5c73000 b5c7a000 r-xp /usr/lib/libembryo.so.1.7.99
b5c82000 b5ca5000 r-xp /usr/lib/libjpeg.so.8.0.2
b5cbd000 b5cd3000 r-xp /usr/lib/libsensor.so.1.1.0
b5cdc000 b5d32000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5d3f000 b5d62000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d6b000 b5db1000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5dba000 b5dcd000 r-xp /usr/lib/libfribidi.so.0.3.1
b5dd5000 b5e25000 r-xp /usr/lib/libfreetype.so.6.8.1
b5e30000 b5e33000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5e3b000 b5e3f000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e47000 b5e4c000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e55000 b5e5f000 r-xp /usr/lib/libXext.so.6.4.0
b5e67000 b5f48000 r-xp /usr/lib/libX11.so.6.3.0
b5f53000 b5f56000 r-xp /usr/lib/libXtst.so.6.1.0
b5f5e000 b5f64000 r-xp /usr/lib/libXrender.so.1.3.0
b5f6c000 b5f71000 r-xp /usr/lib/libXrandr.so.2.2.0
b5f79000 b5f7a000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f83000 b5f8b000 r-xp /usr/lib/libXi.so.6.1.0
b5f8c000 b5f8f000 r-xp /usr/lib/libXfixes.so.3.1.0
b5f97000 b5f99000 r-xp /usr/lib/libXgesture.so.7.0.0
b5fa1000 b5fa3000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5fab000 b5fac000 r-xp /usr/lib/libXdamage.so.1.1.0
b5fb5000 b5fbb000 r-xp /usr/lib/libXcursor.so.1.0.2
b5fc4000 b5fdd000 r-xp /usr/lib/libecore_con.so.1.7.99
b5fe7000 b5fed000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5ff5000 b5ffd000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6005000 b6009000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b6012000 b6028000 r-xp /usr/lib/libefreet.so.1.7.99
b6031000 b603a000 r-xp /usr/lib/libedbus.so.1.7.99
b6042000 b6127000 r-xp /usr/lib/libicuuc.so.51.1
b613c000 b627b000 r-xp /usr/lib/libicui18n.so.51.1
b628b000 b62e7000 r-xp /usr/lib/libedje.so.1.7.99
b62f1000 b6302000 r-xp /usr/lib/libecore_input.so.1.7.99
b630a000 b630f000 r-xp /usr/lib/libecore_file.so.1.7.99
b6317000 b6330000 r-xp /usr/lib/libeet.so.1.7.99
b6341000 b6345000 r-xp /usr/lib/libappcore-common.so.1.1
b634e000 b641a000 r-xp /usr/lib/libevas.so.1.7.99
b643f000 b6460000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6469000 b6498000 r-xp /usr/lib/libecore_x.so.1.7.99
b64a2000 b65d6000 r-xp /usr/lib/libelementary.so.1.7.99
b65ee000 b65ef000 r-xp /usr/lib/libjournal.so.0.1.0
b65f8000 b66c3000 r-xp /usr/lib/libxml2.so.2.7.8
b66d1000 b66e1000 r-xp /lib/libresolv-2.13.so
b66e5000 b66fb000 r-xp /lib/libz.so.1.2.5
b6703000 b6705000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b670d000 b6712000 r-xp /usr/lib/libffi.so.5.0.10
b671b000 b671c000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6724000 b6727000 r-xp /lib/libattr.so.1.1.0
b672f000 b68d7000 r-xp /usr/lib/libcrypto.so.1.0.0
b68f7000 b6911000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b691a000 b6983000 r-xp /lib/libm-2.13.so
b698c000 b69cc000 r-xp /usr/lib/libeina.so.1.7.99
b69d5000 b69dd000 r-xp /usr/lib/libvconf.so.0.2.45
b69e5000 b69e8000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b69f0000 b6a24000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a2d000 b6b01000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b0d000 b6b13000 r-xp /lib/librt-2.13.so
b6b1c000 b6b21000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6b2a000 b6b31000 r-xp /lib/libcrypt-2.13.so
b6b61000 b6b64000 r-xp /lib/libcap.so.2.21
b6b6c000 b6b6e000 r-xp /usr/lib/libiri.so
b6b76000 b6b95000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b9d000 b6bb3000 r-xp /usr/lib/libecore.so.1.7.99
b6bc9000 b6bce000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6bd7000 b6ca7000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6ca8000 b6cb6000 r-xp /usr/lib/libail.so.0.1.0
b6cbe000 b6cd5000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6cde000 b6ce8000 r-xp /lib/libunwind.so.8.0.1
b6d16000 b6e31000 r-xp /lib/libc-2.13.so
b6e3f000 b6e47000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e4f000 b6e79000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e82000 b6e85000 r-xp /usr/lib/libbundle.so.0.1.22
b6e8d000 b6e8f000 r-xp /lib/libdl-2.13.so
b6e98000 b6e9b000 r-xp /usr/lib/libsmack.so.1.0.0
b6ea3000 b6f05000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f0f000 b6f21000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f2a000 b6f3e000 r-xp /lib/libpthread-2.13.so
b6f4b000 b6f4f000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f59000 b6f5b000 r-xp /usr/lib/libdlog.so.0.0.0
b6f63000 b6f6e000 r-xp /usr/lib/libaul.so.0.1.0
b6f78000 b6f7c000 r-xp /usr/lib/libsys-assert.so
b6f85000 b6fa2000 r-xp /lib/ld-2.13.so
b6fab000 b6fb1000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6fb9000 b6fe5000 rw-p [heap]
b6fe5000 b7200000 rw-p [heap]
beed8000 beef9000 rwxp [stack]
End of Maps Information

Callstack Information (PID:4096)
Call Stack Count: 1
 0: (0xb3b66ae4) [/usr/lib/libMali.so] + 0x14ae4
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
op_add(1590) > Cannot create a table. Maybe there is already a table.
04-02 19:24:33.945+0900 D/AIL     ( 4035): ail_desktop.c: _read_desktop_info(962) > Read (org.tizen.other).
04-02 19:24:33.960+0900 D/AIL     ( 4035): ail_desktop.c: _insert_desktop_info(1309) > Add (org.tizen.other).
04-02 19:24:33.970+0900 D/AIL     ( 4035): ail_desktop.c: _send_db_done_noti(1513) > Noti : create:org.tizen.other
04-02 19:24:33.970+0900 D/PKGMGR_PARSER( 4035): pkgmgr_parser.c: pkgmgr_parser_parse_manifest_for_installation(2186) > Creating desktop file Success
04-02 19:24:33.970+0900 D/PKGMGR_PARSER( 4035): pkgmgr_parser.c: pkgmgr_parser_parse_manifest_for_installation(2191) > Free Done
04-02 19:24:33.970+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_package_install(2164) > manifest parsing done successfully.
04-02 19:24:33.980+0900 D/rpm-installer( 4035): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[60]
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(427) > Transaction Begin
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 1 11
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 2 11
04-02 19:24:33.985+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 2210): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.985+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.985+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:33.985+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 2250): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.985+0900 D/MENU_SCREEN( 2250): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:60] for org.tizen.other
04-02 19:24:33.985+0900 D/MENU_SCREEN( 2250): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.985+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 10 1
04-02 19:24:33.985+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 2289): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 4 2
04-02 19:24:33.985+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.985+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 5 2
04-02 19:24:33.985+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 6 2
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(419) > event_cb is called
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.985+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:33.985+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 2291): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.985+0900 D/DATA_PROVIDER_MASTER( 2291): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 60
04-02 19:24:33.985+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.985+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:33.990+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.990+0900 D/PKGMGR  ( 2422): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.990+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.990+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:33.990+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 60
04-02 19:24:33.990+0900 D/PKGMGR  ( 2294): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-02 19:24:33.990+0900 D/QUICKPANEL( 2294): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:60
04-02 19:24:33.990+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:33.990+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.005+0900 E/PKGMGR_CERT( 4035): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(571) > Transaction Commit and End
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/res]
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_make_directory(1490) > mkdir failed. appdir=[/usr/apps/org.tizen.other/shared], errno=[2][No such file or directory]
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared]
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/opt/usr/apps/org.tizen.other/shared/data]
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared/res]
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/tizen-manifest.xml]
04-02 19:24:34.005+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/author-signature.xml]
04-02 19:24:34.010+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/signature1.xml]
04-02 19:24:34.010+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/share/packages/org.tizen.other.xml]
04-02 19:24:34.010+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_make_directory_for_ext(1353) > Directory dose not exist. path: /opt/usr/apps/org.tizen.other/shared/data, errno: 2 (No such file or directory)
04-02 19:24:34.120+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_register_package(64) > [smack] app_install(org.tizen.other), result=[0]
04-02 19:24:34.125+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other, 5, _), result=[0]
04-02 19:24:34.130+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared, 5, _), result=[0]
04-02 19:24:34.130+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/res, 5, _), result=[0]
04-02 19:24:34.130+0900 E/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_get_smack_label_access(627) > Error in getting smack ACCESS label failed. result[-1] (path:[/opt/usr/apps/org.tizen.other/shared/data]))
04-02 19:24:34.135+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_get_group_id(1770) > encoding done, len=[28]
04-02 19:24:34.135+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_apply_smack(1878) > groupid = [QVI9+j1mlN94KHCZf1TZSIDuLJk=] for shared/trusted.
04-02 19:24:34.235+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/trusted, 1, QVI9+j1mlN94KHCZf1TZSIDuLJk=), result=[0]
04-02 19:24:34.235+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/bin, 0, org.tizen.other), result=[0]
04-02 19:24:34.235+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/lib, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/tizen-manifest.xml, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/author-signature.xml, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/signature1.xml, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/share/packages/org.tizen.other.xml, 0, org.tizen.other), result=[0]
04-02 19:24:34.240+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other, 5, _), result=[0]
04-02 19:24:34.245+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-02 19:24:34.245+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-02 19:24:34.245+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/shared, 5, _), result=[0]
04-02 19:24:34.255+0900 D/rpm-installer( 4035): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/internet]
04-02 19:24:34.280+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-02 19:24:34.285+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-02 19:24:34.395+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-02 19:24:34.395+0900 D/rpm-installer( 4035): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-02 19:24:34.395+0900 D/rpm-installer( 4035): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/systemsettings]
04-02 19:24:34.420+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-02 19:24:34.420+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-02 19:24:34.535+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-02 19:24:34.535+0900 D/rpm-installer( 4035): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-02 19:24:34.570+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-02 19:24:34.575+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-02 19:24:34.710+0900 D/rpm-installer( 4035): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-02 19:24:34.710+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_package_install(2194) > permission applying done successfully.
04-02 19:24:34.710+0900 D/PRIVILEGE_INFO( 4035): privilege_info.c: privilege_manager_verify_privilege_list(625) > privilege_info_compare_privilege_level called
04-02 19:24:34.710+0900 D/PRIVILEGE_INFO( 4035): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/internet
04-02 19:24:34.715+0900 D/PRIVILEGE_INFO( 4035): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/systemsettings
04-02 19:24:34.720+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_verify_privilege_list(672) > privilege_manager_verify_privilege_list(PRVMGR_PACKAGE_TYPE_CORE) is ok.
04-02 19:24:34.720+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_package_install(2202) > _coretpk_installer_verify_privilege_list done.
04-02 19:24:34.720+0900 D/rpm-installer( 4035): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[100]
04-02 19:24:34.720+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_package_install(2224) > install status is [2].
04-02 19:24:34.720+0900 D/rpm-installer( 4035): coretpk-installer.c: __post_install_for_mmc(311) > Installed storage is internal.
04-02 19:24:34.720+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_package_install(2231) > _coretpk_installer_package_install is done.
04-02 19:24:34.720+0900 D/rpm-installer( 4035): rpm-vconf-intf.c: _ri_broadcast_status_notification(196) > pkgid=[org.tizen.other], key=[end], val=[ok]
04-02 19:24:34.725+0900 D/rpm-installer( 4035): coretpk-installer.c: _coretpk_installer_prepare_package_install(2695) > success
04-02 19:24:34.725+0900 D/rpm-installer( 4035): rpm-appcore-intf.c: main(224) > sync() start
04-02 19:24:34.725+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.725+0900 D/PKGMGR  ( 2210): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.725+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.725+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.725+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.725+0900 D/PKGMGR  ( 2289): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.725+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.725+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(419) > event_cb is called
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.725+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.725+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.725+0900 D/PKGMGR  ( 2291): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.725+0900 D/DATA_PROVIDER_MASTER( 2291): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 100
04-02 19:24:34.725+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.725+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.730+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.730+0900 D/PKGMGR  ( 2250): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.730+0900 D/MENU_SCREEN( 2250): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:100] for org.tizen.other
04-02 19:24:34.730+0900 D/MENU_SCREEN( 2250): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 100
04-02 19:24:34.730+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.730+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.730+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.730+0900 D/PKGMGR  ( 2422): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.730+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.730+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.730+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / install_percent / 100
04-02 19:24:34.730+0900 D/PKGMGR  ( 2294): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-02 19:24:34.730+0900 D/QUICKPANEL( 2294): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:100
04-02 19:24:34.730+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.730+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.730+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2289): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.735+0900 D/PKGMGR  ( 2289): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.735+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2422): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.735+0900 D/PKGMGR  ( 2422): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.735+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2291): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/DATA_PROVIDER_MASTER( 2291): pkgmgr.c: end_cb(409) > [SECURE_LOG] [org.tizen.other] ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): pkgmgr.c: __operation_callback(419) > event_cb is called
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.735+0900 D/PKGMGR  ( 4033): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.735+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2250): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/MENU_SCREEN( 2250): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [end:ok] for org.tizen.other
04-02 19:24:34.735+0900 D/MENU_SCREEN( 2250): pkgmgr.c: _end(520) > Package(org.tizen.other) : key(install) - val(ok)
04-02 19:24:34.735+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2294): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/QUICKPANEL( 2294): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:end val:ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.735+0900 D/PKGMGR  ( 2294): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.735+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.735+0900 D/PKGMGR  ( 2210): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.735+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.735+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.740+0900 D/PKGMGR  ( 2180): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [install] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280 / coretpk / org.tizen.other / end / ok
04-02 19:24:34.740+0900 D/PKGMGR  ( 2180): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1020474280] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-02 19:24:34.740+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(538) > [SECURE_LOG] pkgid(org.tizen.other), key(end), value(ok)
04-02 19:24:34.740+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(559) > [SECURE_LOG] op(install), value(ok)
04-02 19:24:34.750+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.750+0900 D/PKGMGR  ( 2291): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:34.830+0900 D/MENU_SCREEN( 2250): layout.c: layout_create_package(200) > package org.tizen.other is installed directly
04-02 19:24:34.830+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __app_info_insert_handler(185) > __app_info_insert_handler
04-02 19:24:34.830+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __app_info_insert_handler(388) > [SECURE_LOG] appinfo file:org.tizen.other, comp:ui, type:rpm
04-02 19:24:34.830+0900 D/PKGMGR  ( 2180): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:34.830+0900 D/PKGMGR  ( 2180): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:35.130+0900 D/MENU_SCREEN( 2250): item.c: item_update(594) > Access to file [/opt/usr/apps/org.tizen.other/shared/res/other.png], size[57662]
04-02 19:24:35.150+0900 D/BADGE   ( 2250): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
04-02 19:24:35.170+0900 D/rpm-installer( 4035): rpm-appcore-intf.c: main(226) > sync() end
04-02 19:24:35.175+0900 D/rpm-installer( 4035): rpm-appcore-intf.c: main(245) > ------------------------------------------------
04-02 19:24:35.175+0900 D/rpm-installer( 4035): rpm-appcore-intf.c: main(246) >  [END] rpm-installer: result=[0]
04-02 19:24:35.175+0900 D/rpm-installer( 4035): rpm-appcore-intf.c: main(247) > ------------------------------------------------
04-02 19:24:35.200+0900 D/PKGMGR_SERVER( 4025): pkgmgr-server.c: sighandler(326) > child exit [4035]
04-02 19:24:35.200+0900 D/PKGMGR_SERVER( 4025): pkgmgr-server.c: sighandler(341) > child NORMAL exit [4035]
04-02 19:24:35.200+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-02 19:24:35.200+0900 D/PKGMGR  ( 2250): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-02 19:24:35.505+0900 D/PKGMGR_SERVER( 4025): pkgmgr-server.c: exit_server(724) > exit_server Start
04-02 19:24:35.505+0900 D/PKGMGR_SERVER( 4025): pkgmgr-server.c: main(1516) > Quit main loop.
04-02 19:24:35.505+0900 D/PKGMGR_SERVER( 4025): pkgmgr-server.c: main(1524) > package manager server terminated.
04-02 19:24:42.165+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 0
04-02 19:24:42.170+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-02 19:24:42.205+0900 D/AUL     ( 2180): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 4093, pid = 4095
04-02 19:24:42.240+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-02 19:24:42.265+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1770) > process_pool: false
04-02 19:24:42.265+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-02 19:24:42.265+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-02 19:24:42.265+0900 D/AUL_AMD ( 2180): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-02 19:24:42.265+0900 D/AUL     ( 2180): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-02 19:24:42.265+0900 D/AUL_PAD ( 2217): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-02 19:24:42.265+0900 D/AUL_PAD ( 2217): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-02 19:24:42.275+0900 D/AUL_PAD ( 2217): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 4096 /opt/usr/apps/org.tizen.other/bin/other
04-02 19:24:42.275+0900 D/AUL_PAD ( 4096): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-02 19:24:42.275+0900 D/AUL_PAD ( 2217): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-02 19:24:42.275+0900 D/AUL_PAD ( 4096): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-02 19:24:42.275+0900 D/AUL_PAD ( 4096): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-02 19:24:42.375+0900 D/AUL_PAD ( 4096): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-02 19:24:42.375+0900 D/AUL_PAD ( 4096): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-02 19:24:42.375+0900 D/AUL_PAD ( 4096): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-02 19:24:42.375+0900 D/AUL_PAD ( 4096): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-02 19:24:42.375+0900 D/LAUNCH  ( 4096): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-02 19:24:42.375+0900 D/AUL_PAD ( 2217): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-02 19:24:42.450+0900 I/CAPI_APPFW_APPLICATION( 4096): app_main.c: ui_app_main(697) > app_efl_main
04-02 19:24:42.450+0900 D/LAUNCH  ( 4096): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-02 19:24:42.475+0900 D/AUL_PAD ( 2217): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-02 19:24:42.475+0900 D/AUL_PAD ( 2217): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-02 19:24:42.475+0900 D/AUL_PAD ( 2217): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-02 19:24:42.475+0900 D/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-02 19:24:42.475+0900 E/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(264) > access
04-02 19:24:42.475+0900 D/RESOURCED( 2382): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-02 19:24:42.475+0900 D/RESOURCED( 2382): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2180
04-02 19:24:42.475+0900 D/RESOURCED( 2382): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 4096, type 4 
04-02 19:24:42.475+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 4096
04-02 19:24:42.475+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-02 19:24:42.475+0900 D/RESOURCED( 2382): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 4096
04-02 19:24:42.475+0900 D/RESOURCED( 2382): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-02 19:24:42.475+0900 D/RESOURCED( 2382): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 4096
04-02 19:24:42.475+0900 D/RESOURCED( 2382): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 4096
04-02 19:24:42.475+0900 D/RESOURCED( 2382): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-02 19:24:42.475+0900 D/RESOURCED( 2382): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-02 19:24:42.480+0900 D/RESOURCED( 2382): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-02 19:24:42.480+0900 D/RESOURCED( 2382): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-02 19:24:42.480+0900 D/RESOURCED( 2382): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-02 19:24:42.480+0900 D/RESOURCED( 2382): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-02 19:24:42.480+0900 E/RESOURCED( 2382): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 575
04-02 19:24:42.480+0900 D/RESOURCED( 2382): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-02 19:24:42.495+0900 D/APP_CORE( 4096): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-02 19:24:42.505+0900 D/AUL     ( 4096): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 4096 is org.tizen.other
04-02 19:24:42.505+0900 D/APP_CORE( 4096): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-02 19:24:42.510+0900 D/APP_CORE( 4096): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-02 19:24:42.510+0900 D/AUL     ( 4096): app_sock.c: __create_server_sock(135) > pg path - already exists
04-02 19:24:42.510+0900 D/LAUNCH  ( 4096): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-02 19:24:42.510+0900 I/CAPI_APPFW_APPLICATION( 4096): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-02 19:24:42.685+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=4096
04-02 19:24:42.750+0900 D/indicator( 2229): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-02 19:24:42.755+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2800003"
04-02 19:24:42.755+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-02 19:24:42.755+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2800003"
04-02 19:24:42.755+0900 D/indicator( 2229): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-02 19:24:42.770+0900 F/socket.io( 4096): thread_start
04-02 19:24:42.770+0900 F/socket.io( 4096): finish 0
04-02 19:24:42.770+0900 D/LAUNCH  ( 4096): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-02 19:24:42.770+0900 D/APP_CORE( 4096): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-02 19:24:42.770+0900 D/APP_CORE( 4096): appcore.c: __aul_handler(423) > [APP 4096]     AUL event: AUL_START
04-02 19:24:42.770+0900 D/APP_CORE( 4096): appcore-efl.c: __do_app(470) > [APP 4096] Event: RESET State: CREATED
04-02 19:24:42.770+0900 D/APP_CORE( 4096): appcore-efl.c: __do_app(496) > [APP 4096] RESET
04-02 19:24:42.770+0900 D/LAUNCH  ( 4096): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-02 19:24:42.770+0900 I/CAPI_APPFW_APPLICATION( 4096): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-02 19:24:42.770+0900 D/APP_SVC ( 4096): appsvc.c: __set_bundle(161) > __set_bundle
04-02 19:24:42.770+0900 D/LAUNCH  ( 4096): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-02 19:24:42.790+0900 I/APP_CORE( 4096): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-02 19:24:42.790+0900 I/APP_CORE( 4096): appcore-efl.c: __do_app(509) > [APP 4096] Initial Launching, call the resume_cb
04-02 19:24:42.790+0900 I/CAPI_APPFW_APPLICATION( 4096): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-02 19:24:42.790+0900 D/APP_CORE( 4096): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-02 19:24:42.795+0900 D/APP_CORE( 4096): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2800003
04-02 19:24:42.795+0900 D/APP_CORE( 4096): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2800003
04-02 19:24:42.810+0900 E/socket.io( 4096): 566: Connected.
04-02 19:24:42.815+0900 E/socket.io( 4096): 554: On handshake, sid
04-02 19:24:42.815+0900 E/socket.io( 4096): 651: Received Message type(connect)
04-02 19:24:42.815+0900 E/socket.io( 4096): 489: On Connected
04-02 19:24:42.820+0900 F/sio_packet( 4096): accept()
04-02 19:24:42.820+0900 E/socket.io( 4096): 743: encoded paylod length: 13
04-02 19:24:42.820+0900 E/socket.io( 4096): 800: ping exit, con is expired? 0, ec: Operation canceled
04-02 19:24:42.890+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:42.890+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:42.890+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-02 19:24:42.895+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:42.940+0900 D/APP_CORE( 2250): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-02 19:24:42.940+0900 D/APP_CORE( 2250): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-02 19:24:42.940+0900 D/APP_CORE( 2250): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-02 19:24:42.940+0900 D/APP_CORE( 2250): appcore-efl.c: __do_app(470) > [APP 2250] Event: PAUSE State: RUNNING
04-02 19:24:42.940+0900 D/APP_CORE( 2250): appcore-efl.c: __do_app(538) > [APP 2250] PAUSE
04-02 19:24:42.940+0900 I/CAPI_APPFW_APPLICATION( 2250): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-02 19:24:42.940+0900 D/MENU_SCREEN( 2250): menu_screen.c: _pause_cb(538) > Pause start
04-02 19:24:42.940+0900 D/APP_CORE( 2250): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-02 19:24:42.940+0900 E/APP_CORE( 2250): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-02 19:24:42.945+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:42.945+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:42.945+0900 D/RESOURCED( 2382): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2250, type = 2
04-02 19:24:42.950+0900 D/RESOURCED( 2382): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 4096, type = 0
04-02 19:24:42.950+0900 D/AUL_AMD ( 2180): amd_launch.c: __e17_status_handler(1888) > pid(4096) status(3)
04-02 19:24:42.950+0900 D/RESOURCED( 2382): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 4096
04-02 19:24:42.950+0900 I/RESOURCED( 2382): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 4096, oom : 200
04-02 19:24:42.950+0900 E/RESOURCED( 2382): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-02 19:24:42.965+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 19:24:42.965+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:42.965+0900 D/APP_CORE( 4096): appcore.c: __prt_ltime(183) > [APP 4096] first idle after reset: 803 msec
04-02 19:24:42.965+0900 D/APP_CORE( 4096): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2800003 fully_obscured 0
04-02 19:24:42.965+0900 D/APP_CORE( 4096): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-02 19:24:42.965+0900 D/APP_CORE( 4096): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-02 19:24:42.970+0900 D/APP_CORE( 4096): appcore-efl.c: __do_app(470) > [APP 4096] Event: RESUME State: RUNNING
04-02 19:24:42.970+0900 D/LAUNCH  ( 4096): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-02 19:24:42.970+0900 D/LAUNCH  ( 4096): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-02 19:24:42.970+0900 D/LAUNCH  ( 4096): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-02 19:24:42.970+0900 D/APP_CORE( 4096): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-02 19:24:42.970+0900 E/APP_CORE( 4096): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-02 19:24:42.995+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:42.995+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:43.005+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 19:24:43.005+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.035+0900 D/RESOURCED( 2382): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
04-02 19:24:43.050+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.050+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:43.055+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 19:24:43.060+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.110+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.110+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:43.120+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 19:24:43.125+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.175+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.175+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:43.185+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 19:24:43.185+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.225+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.225+0900 F/get_binary( 4096): in get binary_message()...
04-02 19:24:43.235+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4096): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 19:24:43.240+0900 E/socket.io( 4096): 669: Received Message type(Event)
04-02 19:24:43.415+0900 W/CRASH_MANAGER( 4109): worker.c: worker_job(1189) > 11040966f7468142797028
