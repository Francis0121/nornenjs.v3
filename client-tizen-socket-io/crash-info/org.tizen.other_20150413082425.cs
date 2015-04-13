S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 11608
Date: 2015-04-13 08:24:25+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0x78c7c73f

Register Information
r0   = 0x78c7c73f, r1   = 0xb0e0d0ac
r2   = 0xc7e6f693, r3   = 0x000000ff
r4   = 0xb7153178, r5   = 0xb712c918
r6   = 0xb708c238, r7   = 0xbec65e70
r8   = 0x00000000, r9   = 0xb715b558
r10  = 0xb715cb98, fp   = 0x00000001
ip   = 0xb6d3f9a4, sp   = 0xbec65e70
lr   = 0xb3e6df53, pc   = 0xb6d3f9b4
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    370256 KB
Buffers:     69928 KB
Cached:     147932 KB
VmPeak:     146648 KB
VmSize:     119648 KB
VmLck:           0 KB
VmHWM:       16240 KB
VmRSS:       16240 KB
VmData:      63672 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       25068 KB
VmPTE:          86 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 11608 TID = 11608
11608 11612 11613 11614 11615 11616 11617 11618 

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
b6f9c000 b7221000 rw-p [heap]
bec46000 bec67000 rwxp [stack]
End of Maps Information

Callstack Information (PID:11608)
Call Stack Count: 20
 0: strcpy + 0x10 (0xb6d3f9b4) [/lib/libc.so.6] + 0x729b4
 1: texture_getter + 0x2e (0xb3e6df53) [/opt/usr/apps/org.tizen.other/bin/other] + 0x8cf53
 2: genTex + 0xe (0xb3dea3fb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x93fb
 3: _btn_clicked_cb + 0x12 (0xb3dea763) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9763
 4: evas_object_smart_callback_call + 0x88 (0xb633c96d) [/usr/lib/libevas.so.1] + 0x3796d
 5: (0xb62858a5) [/usr/lib/libedje.so.1] + 0x438a5
 6: (0xb628a2d1) [/usr/lib/libedje.so.1] + 0x482d1
 7: (0xb6286201) [/usr/lib/libedje.so.1] + 0x44201
 8: (0xb62865b3) [/usr/lib/libedje.so.1] + 0x445b3
 9: (0xb62866ed) [/usr/lib/libedje.so.1] + 0x446ed
10: (0xb6b5e3cd) [/usr/lib/libecore.so.1] + 0xa3cd
11: (0xb6b5be77) [/usr/lib/libecore.so.1] + 0x7e77
12: (0xb6b5f443) [/usr/lib/libecore.so.1] + 0xb443
13: ecore_main_loop_begin + 0x30 (0xb6b5f851) [/usr/lib/libecore.so.1] + 0xb851
14: appcore_efl_main + 0x2a2 (0xb6f046f3) [/usr/lib/libappcore-efl.so.1] + 0x26f3
15: ui_app_main + 0xb0 (0xb451ec79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
16: main + 0x198 (0xb3deafe9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9fe9
17: (0xb6f64c6f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c6f
18: __libc_start_main + 0x114 (0xb6ce482c) [/lib/libc.so.6] + 0x1782c
19: (0xb6f6535c) [/opt/usr/apps/org.tizen.other/bin/other] + 0x335c
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
608): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-13 08:23:57.995+0900 D/RESOURCED( 2372): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-13 08:23:57.995+0900 D/RESOURCED( 2372): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-13 08:23:57.995+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 08:23:57.995+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 08:23:57.995+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 08:23:57.995+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 08:23:57.995+0900 E/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 522
04-13 08:23:57.995+0900 D/RESOURCED( 2372): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-13 08:23:58.000+0900 D/AUL     (11608): app_sock.c: __create_server_sock(135) > pg path - already exists
04-13 08:23:58.000+0900 D/LAUNCH  (11608): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-13 08:23:58.000+0900 I/CAPI_APPFW_APPLICATION(11608): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-13 08:23:58.190+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=11608
04-13 08:23:58.240+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-13 08:23:58.245+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
04-13 08:23:58.245+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-13 08:23:58.255+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
04-13 08:23:58.255+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-13 08:23:58.270+0900 D/LAUNCH  (11608): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-13 08:23:58.270+0900 D/APP_CORE(11608): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-13 08:23:58.270+0900 D/APP_CORE(11608): appcore.c: __aul_handler(423) > [APP 11608]     AUL event: AUL_START
04-13 08:23:58.270+0900 D/APP_CORE(11608): appcore-efl.c: __do_app(470) > [APP 11608] Event: RESET State: CREATED
04-13 08:23:58.270+0900 D/APP_CORE(11608): appcore-efl.c: __do_app(496) > [APP 11608] RESET
04-13 08:23:58.270+0900 D/LAUNCH  (11608): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-13 08:23:58.270+0900 I/CAPI_APPFW_APPLICATION(11608): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-13 08:23:58.270+0900 D/APP_SVC (11608): appsvc.c: __set_bundle(161) > __set_bundle
04-13 08:23:58.270+0900 F/socket.io(11608): thread_start
04-13 08:23:58.270+0900 F/socket.io(11608): finish 0
04-13 08:23:58.270+0900 D/LAUNCH  (11608): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-13 08:23:58.275+0900 F/socket.io(11608): Connect Start
04-13 08:23:58.275+0900 F/socket.io(11608): Set ConnectListener
04-13 08:23:58.275+0900 F/socket.io(11608): Set ClosetListener
04-13 08:23:58.275+0900 F/socket.io(11608): Set FaileListener
04-13 08:23:58.275+0900 F/socket.io(11608): Connect
04-13 08:23:58.275+0900 F/socket.io(11608): Lock
04-13 08:23:58.275+0900 F/socket.io(11608): !!!
04-13 08:23:58.295+0900 I/APP_CORE(11608): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-13 08:23:58.295+0900 I/APP_CORE(11608): appcore-efl.c: __do_app(509) > [APP 11608] Initial Launching, call the resume_cb
04-13 08:23:58.295+0900 I/CAPI_APPFW_APPLICATION(11608): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-13 08:23:58.295+0900 D/APP_CORE(11608): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-13 08:23:58.295+0900 D/APP_CORE(11608): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600003
04-13 08:23:58.295+0900 D/APP_CORE(11608): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600003
04-13 08:23:58.375+0900 E/socket.io(11608): 566: Connected.
04-13 08:23:58.375+0900 D/sio_packet(11608): from json
04-13 08:23:58.375+0900 D/value.IsObject()(11608): if binary from json
04-13 08:23:58.375+0900 D/sio_packet(11608): from json
04-13 08:23:58.375+0900 D/sio_packet(11608): from json
04-13 08:23:58.375+0900 D/value.IsArray()(11608): if arr from json
04-13 08:23:58.375+0900 D/sio_packet(11608): from json
04-13 08:23:58.375+0900 D/sio_packet(11608): IsInt64
04-13 08:23:58.375+0900 D/sio_packet(11608): from json
04-13 08:23:58.375+0900 D/sio_packet(11608): IsInt64
04-13 08:23:58.375+0900 E/socket.io(11608): 554: On handshake, sid
04-13 08:23:58.375+0900 E/socket.io(11608): 651: Received Message type(connect)
04-13 08:23:58.375+0900 E/socket.io(11608): 489: On Connected
04-13 08:23:58.375+0900 F/socket.io(11608): unlock
04-13 08:23:58.375+0900 F/socket.io(11608): emit connectMessage
04-13 08:23:58.375+0900 F/sio_packet(11608): accept()
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_message
04-13 08:23:58.375+0900 F/sio_packet(11608): arr, but we will use it like binary
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_array_message()
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_message
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_message
04-13 08:23:58.375+0900 E/socket.io(11608): 743: encoded paylod length: 63
04-13 08:23:58.375+0900 F/socket.io(11608): bind connectMessage
04-13 08:23:58.375+0900 F/socket.io(11608): emit connectMessage
04-13 08:23:58.375+0900 F/sio_packet(11608): accept()
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_message
04-13 08:23:58.375+0900 F/sio_packet(11608): arr, but we will use it like binary
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_array_message()
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_message
04-13 08:23:58.375+0900 F/sio_packet(11608): start accept_message
04-13 08:23:58.375+0900 E/socket.io(11608): 743: encoded paylod length: 13
04-13 08:23:58.375+0900 F/socket.io(11608): close 11608
04-13 08:23:58.375+0900 E/socket.io(11608): 800: ping exit, con is expired? 0, ec: Operation canceled
04-13 08:23:58.375+0900 E/socket.io(11608): 800: ping exit, con is expired? 0, ec: Operation canceled
04-13 08:23:58.380+0900 D/sio_packet(11608): from json
04-13 08:23:58.380+0900 D/value.IsArray()(11608): if arr from json
04-13 08:23:58.380+0900 D/sio_packet(11608): from json
04-13 08:23:58.380+0900 D/sio_packet(11608): from json
04-13 08:23:58.380+0900 D/value.IsObject()(11608): if binary from json
04-13 08:23:58.380+0900 D/sio_packet(11608): from json
04-13 08:23:58.380+0900 D/sio_packet(11608): from json
04-13 08:23:58.380+0900 D/sio_packet(11608): from json
04-13 08:23:58.380+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:23:58.380+0900 F/socket.io(11608): bind_event [connectMessage] 11608
04-13 08:23:58.380+0900 F/socket.io(11608): connectMessage :: 
04-13 08:23:58.405+0900 F/other   (11608): success!!!! -1223401960 -1223401960 
04-13 08:23:58.440+0900 D/APP_CORE( 2245): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-13 08:23:58.440+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-13 08:23:58.445+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-13 08:23:58.445+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(470) > [APP 2245] Event: PAUSE State: RUNNING
04-13 08:23:58.445+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(538) > [APP 2245] PAUSE
04-13 08:23:58.445+0900 I/CAPI_APPFW_APPLICATION( 2245): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-13 08:23:58.445+0900 D/MENU_SCREEN( 2245): menu_screen.c: _pause_cb(538) > Pause start
04-13 08:23:58.445+0900 D/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 08:23:58.445+0900 E/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 08:23:58.445+0900 D/APP_CORE(11608): appcore.c: __prt_ltime(183) > [APP 11608] first idle after reset: 582 msec
04-13 08:23:58.445+0900 D/APP_CORE(11608): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 0
04-13 08:23:58.445+0900 D/APP_CORE(11608): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-13 08:23:58.445+0900 D/APP_CORE(11608): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-13 08:23:58.445+0900 D/APP_CORE(11608): appcore-efl.c: __do_app(470) > [APP 11608] Event: RESUME State: RUNNING
04-13 08:23:58.445+0900 D/LAUNCH  (11608): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-13 08:23:58.445+0900 D/LAUNCH  (11608): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-13 08:23:58.445+0900 D/LAUNCH  (11608): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-13 08:23:58.445+0900 D/APP_CORE(11608): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 08:23:58.445+0900 E/APP_CORE(11608): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 08:23:58.450+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2245, type = 2
04-13 08:23:58.450+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 11608, type = 0
04-13 08:23:58.450+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(11608) status(3)
04-13 08:23:58.450+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 11608
04-13 08:23:58.450+0900 I/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 11608, oom : 200
04-13 08:23:58.450+0900 E/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-13 08:23:59.000+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.other /opt/usr/apps/org.tizen.other/bin/other
04-13 08:23:59.000+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
04-13 08:23:59.020+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
04-13 08:23:59.195+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-13 08:23:59.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
04-13 08:23:59.450+0900 D/sio_packet(11608): from json
04-13 08:23:59.450+0900 D/value.IsArray()(11608): if arr from json
04-13 08:23:59.450+0900 D/sio_packet(11608): from json
04-13 08:23:59.450+0900 D/sio_packet(11608): from json
04-13 08:23:59.450+0900 D/value.IsObject()(11608): if binary from json
04-13 08:23:59.450+0900 D/sio_packet(11608): from json
04-13 08:23:59.450+0900 D/value.IsObject()(11608): if binary from json
04-13 08:23:59.450+0900 D/sio_packet(11608): from json
04-13 08:23:59.450+0900 D/sio_packet(11608): IsInt64
04-13 08:23:59.450+0900 D/sio_packet(11608): from json
04-13 08:23:59.450+0900 D/value.IsObject()(11608): if binary from json
04-13 08:23:59.450+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:23:59.450+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:23:59.450+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:23:59.450+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:23:59.450+0900 F/socket.io(11608): key size : 9312
04-13 08:23:59.450+0900 F/get_binary(11608): in get binary_message()...
04-13 08:23:59.450+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 08:24 5 HH:mm"
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 08:24"
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 08&#x2236;24"
04-13 08:24:00.250+0900 D/indicator( 2218): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 366992 Time: <font_size=34>08&#x2236;24</font_size></font>"
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/sio_packet(11608): IsInt64
04-13 08:24:00.485+0900 D/sio_packet(11608): from json
04-13 08:24:00.485+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:00.485+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:00.485+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:00.485+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:00.485+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:00.485+0900 F/socket.io(11608): key size : 9312
04-13 08:24:00.485+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:00.485+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:01.600+0900 D/sio_packet(11608): from json
04-13 08:24:01.600+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:01.600+0900 D/sio_packet(11608): from json
04-13 08:24:01.600+0900 D/sio_packet(11608): from json
04-13 08:24:01.600+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:01.600+0900 D/sio_packet(11608): from json
04-13 08:24:01.600+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:01.600+0900 D/sio_packet(11608): from json
04-13 08:24:01.600+0900 D/sio_packet(11608): IsInt64
04-13 08:24:01.600+0900 D/sio_packet(11608): from json
04-13 08:24:01.600+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:01.600+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:01.600+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:01.600+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:01.600+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:01.600+0900 F/socket.io(11608): key size : 9312
04-13 08:24:01.600+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:01.600+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:02.420+0900 D/sio_packet(11608): from json
04-13 08:24:02.420+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:02.420+0900 D/sio_packet(11608): from json
04-13 08:24:02.420+0900 D/sio_packet(11608): from json
04-13 08:24:02.420+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:02.420+0900 D/sio_packet(11608): from json
04-13 08:24:02.420+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:02.510+0900 D/sio_packet(11608): from json
04-13 08:24:02.510+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:02.510+0900 D/sio_packet(11608): from json
04-13 08:24:02.510+0900 D/sio_packet(11608): from json
04-13 08:24:02.510+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:02.510+0900 D/sio_packet(11608): from json
04-13 08:24:02.510+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:02.510+0900 D/sio_packet(11608): from json
04-13 08:24:02.510+0900 D/sio_packet(11608): IsInt64
04-13 08:24:02.510+0900 D/sio_packet(11608): from json
04-13 08:24:02.510+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:02.510+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:02.510+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:02.510+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:02.510+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:02.510+0900 F/socket.io(11608): key size : 9312
04-13 08:24:02.510+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:02.510+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:03.515+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(470) > [APP 2245] Event: MEM_FLUSH State: PAUSED
04-13 08:24:03.545+0900 D/sio_packet(11608): from json
04-13 08:24:03.545+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:03.545+0900 D/sio_packet(11608): from json
04-13 08:24:03.545+0900 D/sio_packet(11608): from json
04-13 08:24:03.545+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:03.545+0900 D/sio_packet(11608): from json
04-13 08:24:03.545+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:03.545+0900 D/sio_packet(11608): from json
04-13 08:24:03.545+0900 D/sio_packet(11608): IsInt64
04-13 08:24:03.545+0900 D/sio_packet(11608): from json
04-13 08:24:03.545+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:03.545+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:03.545+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:03.545+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:03.545+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:03.545+0900 F/socket.io(11608): key size : 9312
04-13 08:24:03.545+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:03.545+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:04.425+0900 D/sio_packet(11608): from json
04-13 08:24:04.425+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:04.425+0900 D/sio_packet(11608): from json
04-13 08:24:04.425+0900 D/sio_packet(11608): from json
04-13 08:24:04.425+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:04.425+0900 D/sio_packet(11608): from json
04-13 08:24:04.425+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:04.570+0900 D/sio_packet(11608): from json
04-13 08:24:04.570+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:04.570+0900 D/sio_packet(11608): from json
04-13 08:24:04.570+0900 D/sio_packet(11608): from json
04-13 08:24:04.570+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:04.570+0900 D/sio_packet(11608): from json
04-13 08:24:04.570+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:04.570+0900 D/sio_packet(11608): from json
04-13 08:24:04.570+0900 D/sio_packet(11608): IsInt64
04-13 08:24:04.570+0900 D/sio_packet(11608): from json
04-13 08:24:04.570+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:04.570+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:04.570+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:04.570+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:04.570+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:04.570+0900 F/socket.io(11608): key size : 9312
04-13 08:24:04.570+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:04.570+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:05.695+0900 D/sio_packet(11608): from json
04-13 08:24:05.695+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:05.695+0900 D/sio_packet(11608): from json
04-13 08:24:05.695+0900 D/sio_packet(11608): from json
04-13 08:24:05.695+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:05.695+0900 D/sio_packet(11608): from json
04-13 08:24:05.695+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:05.695+0900 D/sio_packet(11608): from json
04-13 08:24:05.695+0900 D/sio_packet(11608): IsInt64
04-13 08:24:05.695+0900 D/sio_packet(11608): from json
04-13 08:24:05.695+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:05.695+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:05.695+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:05.695+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:05.695+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:05.695+0900 F/socket.io(11608): key size : 9312
04-13 08:24:05.695+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:05.695+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:06.415+0900 D/sio_packet(11608): from json
04-13 08:24:06.415+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:06.415+0900 D/sio_packet(11608): from json
04-13 08:24:06.415+0900 D/sio_packet(11608): from json
04-13 08:24:06.415+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:06.415+0900 D/sio_packet(11608): from json
04-13 08:24:06.415+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:06.720+0900 D/sio_packet(11608): from json
04-13 08:24:06.720+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:06.720+0900 D/sio_packet(11608): from json
04-13 08:24:06.720+0900 D/sio_packet(11608): from json
04-13 08:24:06.720+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:06.720+0900 D/sio_packet(11608): from json
04-13 08:24:06.720+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:06.720+0900 D/sio_packet(11608): from json
04-13 08:24:06.720+0900 D/sio_packet(11608): IsInt64
04-13 08:24:06.720+0900 D/sio_packet(11608): from json
04-13 08:24:06.720+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:06.720+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:06.720+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:06.720+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:06.720+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:06.720+0900 F/socket.io(11608): key size : 9312
04-13 08:24:06.720+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:06.720+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:07.740+0900 D/sio_packet(11608): from json
04-13 08:24:07.740+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:07.740+0900 D/sio_packet(11608): from json
04-13 08:24:07.740+0900 D/sio_packet(11608): from json
04-13 08:24:07.740+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:07.740+0900 D/sio_packet(11608): from json
04-13 08:24:07.740+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:07.740+0900 D/sio_packet(11608): from json
04-13 08:24:07.740+0900 D/sio_packet(11608): IsInt64
04-13 08:24:07.740+0900 D/sio_packet(11608): from json
04-13 08:24:07.740+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:07.740+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:07.740+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:07.740+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:07.740+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:07.740+0900 F/socket.io(11608): key size : 9312
04-13 08:24:07.740+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:07.740+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:08.460+0900 D/sio_packet(11608): from json
04-13 08:24:08.460+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:08.460+0900 D/sio_packet(11608): from json
04-13 08:24:08.460+0900 D/sio_packet(11608): from json
04-13 08:24:08.460+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:08.460+0900 D/sio_packet(11608): from json
04-13 08:24:08.460+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:08.770+0900 D/sio_packet(11608): from json
04-13 08:24:08.770+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:08.770+0900 D/sio_packet(11608): from json
04-13 08:24:08.770+0900 D/sio_packet(11608): from json
04-13 08:24:08.770+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:08.770+0900 D/sio_packet(11608): from json
04-13 08:24:08.770+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:08.770+0900 D/sio_packet(11608): from json
04-13 08:24:08.770+0900 D/sio_packet(11608): IsInt64
04-13 08:24:08.770+0900 D/sio_packet(11608): from json
04-13 08:24:08.770+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:08.770+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:08.770+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:08.770+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:08.770+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:08.770+0900 F/socket.io(11608): key size : 9312
04-13 08:24:08.770+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:08.770+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:09.790+0900 D/sio_packet(11608): from json
04-13 08:24:09.790+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:09.790+0900 D/sio_packet(11608): from json
04-13 08:24:09.790+0900 D/sio_packet(11608): from json
04-13 08:24:09.790+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:09.790+0900 D/sio_packet(11608): from json
04-13 08:24:09.790+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:09.790+0900 D/sio_packet(11608): from json
04-13 08:24:09.790+0900 D/sio_packet(11608): IsInt64
04-13 08:24:09.790+0900 D/sio_packet(11608): from json
04-13 08:24:09.790+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:09.790+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:09.790+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:09.790+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:09.790+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:09.790+0900 F/socket.io(11608): key size : 9312
04-13 08:24:09.790+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:09.790+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:10.515+0900 D/sio_packet(11608): from json
04-13 08:24:10.515+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:10.515+0900 D/sio_packet(11608): from json
04-13 08:24:10.515+0900 D/sio_packet(11608): from json
04-13 08:24:10.515+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:10.515+0900 D/sio_packet(11608): from json
04-13 08:24:10.515+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:10.820+0900 D/sio_packet(11608): from json
04-13 08:24:10.820+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:10.820+0900 D/sio_packet(11608): from json
04-13 08:24:10.820+0900 D/sio_packet(11608): from json
04-13 08:24:10.820+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:10.820+0900 D/sio_packet(11608): from json
04-13 08:24:10.820+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:10.820+0900 D/sio_packet(11608): from json
04-13 08:24:10.820+0900 D/sio_packet(11608): IsInt64
04-13 08:24:10.820+0900 D/sio_packet(11608): from json
04-13 08:24:10.820+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:10.820+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:10.820+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:10.820+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:10.820+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:10.820+0900 F/socket.io(11608): key size : 9312
04-13 08:24:10.820+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:10.820+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:11.845+0900 D/sio_packet(11608): from json
04-13 08:24:11.845+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:11.845+0900 D/sio_packet(11608): from json
04-13 08:24:11.845+0900 D/sio_packet(11608): from json
04-13 08:24:11.845+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:11.845+0900 D/sio_packet(11608): from json
04-13 08:24:11.845+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:11.845+0900 D/sio_packet(11608): from json
04-13 08:24:11.845+0900 D/sio_packet(11608): IsInt64
04-13 08:24:11.845+0900 D/sio_packet(11608): from json
04-13 08:24:11.845+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:11.845+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:11.845+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:11.845+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:11.845+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:11.845+0900 F/socket.io(11608): key size : 9312
04-13 08:24:11.845+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:11.845+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:12.455+0900 D/sio_packet(11608): from json
04-13 08:24:12.455+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:12.455+0900 D/sio_packet(11608): from json
04-13 08:24:12.455+0900 D/sio_packet(11608): from json
04-13 08:24:12.455+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:12.455+0900 D/sio_packet(11608): from json
04-13 08:24:12.455+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:12.875+0900 D/sio_packet(11608): from json
04-13 08:24:12.875+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:12.875+0900 D/sio_packet(11608): from json
04-13 08:24:12.875+0900 D/sio_packet(11608): from json
04-13 08:24:12.875+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:12.875+0900 D/sio_packet(11608): from json
04-13 08:24:12.875+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:12.875+0900 D/sio_packet(11608): from json
04-13 08:24:12.875+0900 D/sio_packet(11608): IsInt64
04-13 08:24:12.875+0900 D/sio_packet(11608): from json
04-13 08:24:12.875+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:12.875+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:12.875+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:12.875+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:12.875+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:12.875+0900 F/socket.io(11608): key size : 9312
04-13 08:24:12.875+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:12.875+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:13.890+0900 D/sio_packet(11608): from json
04-13 08:24:13.890+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:13.890+0900 D/sio_packet(11608): from json
04-13 08:24:13.890+0900 D/sio_packet(11608): from json
04-13 08:24:13.890+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:13.890+0900 D/sio_packet(11608): from json
04-13 08:24:13.890+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:13.890+0900 D/sio_packet(11608): from json
04-13 08:24:13.890+0900 D/sio_packet(11608): IsInt64
04-13 08:24:13.890+0900 D/sio_packet(11608): from json
04-13 08:24:13.890+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:13.890+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:13.890+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:13.890+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:13.890+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:13.890+0900 F/socket.io(11608): key size : 9312
04-13 08:24:13.890+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:13.890+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:14.510+0900 D/sio_packet(11608): from json
04-13 08:24:14.510+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:14.510+0900 D/sio_packet(11608): from json
04-13 08:24:14.510+0900 D/sio_packet(11608): from json
04-13 08:24:14.510+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:14.510+0900 D/sio_packet(11608): from json
04-13 08:24:14.510+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:14.915+0900 D/sio_packet(11608): from json
04-13 08:24:14.915+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:14.915+0900 D/sio_packet(11608): from json
04-13 08:24:14.915+0900 D/sio_packet(11608): from json
04-13 08:24:14.915+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:14.915+0900 D/sio_packet(11608): from json
04-13 08:24:14.915+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:14.915+0900 D/sio_packet(11608): from json
04-13 08:24:14.915+0900 D/sio_packet(11608): IsInt64
04-13 08:24:14.915+0900 D/sio_packet(11608): from json
04-13 08:24:14.915+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:14.915+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:14.915+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:14.915+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:14.915+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:14.915+0900 F/socket.io(11608): key size : 9312
04-13 08:24:14.915+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:14.915+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:16.040+0900 D/sio_packet(11608): from json
04-13 08:24:16.040+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:16.040+0900 D/sio_packet(11608): from json
04-13 08:24:16.040+0900 D/sio_packet(11608): from json
04-13 08:24:16.040+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:16.040+0900 D/sio_packet(11608): from json
04-13 08:24:16.040+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:16.040+0900 D/sio_packet(11608): from json
04-13 08:24:16.040+0900 D/sio_packet(11608): IsInt64
04-13 08:24:16.040+0900 D/sio_packet(11608): from json
04-13 08:24:16.040+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:16.040+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:16.040+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:16.040+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:16.040+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:16.040+0900 F/socket.io(11608): key size : 9312
04-13 08:24:16.040+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:16.040+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:16.445+0900 D/sio_packet(11608): from json
04-13 08:24:16.445+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:16.445+0900 D/sio_packet(11608): from json
04-13 08:24:16.450+0900 D/sio_packet(11608): from json
04-13 08:24:16.450+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:16.450+0900 D/sio_packet(11608): from json
04-13 08:24:16.450+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:17.065+0900 D/sio_packet(11608): from json
04-13 08:24:17.065+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:17.065+0900 D/sio_packet(11608): from json
04-13 08:24:17.065+0900 D/sio_packet(11608): from json
04-13 08:24:17.065+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:17.065+0900 D/sio_packet(11608): from json
04-13 08:24:17.065+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:17.065+0900 D/sio_packet(11608): from json
04-13 08:24:17.065+0900 D/sio_packet(11608): IsInt64
04-13 08:24:17.065+0900 D/sio_packet(11608): from json
04-13 08:24:17.065+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:17.065+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:17.065+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:17.065+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:17.065+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:17.065+0900 F/socket.io(11608): key size : 9312
04-13 08:24:17.065+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:17.065+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:18.085+0900 D/sio_packet(11608): from json
04-13 08:24:18.090+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:18.090+0900 D/sio_packet(11608): from json
04-13 08:24:18.090+0900 D/sio_packet(11608): from json
04-13 08:24:18.090+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:18.090+0900 D/sio_packet(11608): from json
04-13 08:24:18.090+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:18.090+0900 D/sio_packet(11608): from json
04-13 08:24:18.090+0900 D/sio_packet(11608): IsInt64
04-13 08:24:18.090+0900 D/sio_packet(11608): from json
04-13 08:24:18.090+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:18.090+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:18.090+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:18.090+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:18.090+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:18.090+0900 F/socket.io(11608): key size : 9312
04-13 08:24:18.090+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:18.090+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:18.335+0900 D/RESOURCED( 2372): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
04-13 08:24:18.495+0900 D/sio_packet(11608): from json
04-13 08:24:18.495+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:18.495+0900 D/sio_packet(11608): from json
04-13 08:24:18.495+0900 D/sio_packet(11608): from json
04-13 08:24:18.495+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:18.495+0900 D/sio_packet(11608): from json
04-13 08:24:18.495+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:19.110+0900 D/sio_packet(11608): from json
04-13 08:24:19.110+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:19.110+0900 D/sio_packet(11608): from json
04-13 08:24:19.110+0900 D/sio_packet(11608): from json
04-13 08:24:19.110+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:19.110+0900 D/sio_packet(11608): from json
04-13 08:24:19.110+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:19.110+0900 D/sio_packet(11608): from json
04-13 08:24:19.110+0900 D/sio_packet(11608): IsInt64
04-13 08:24:19.110+0900 D/sio_packet(11608): from json
04-13 08:24:19.110+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:19.110+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:19.110+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:19.110+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:19.110+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:19.110+0900 F/socket.io(11608): key size : 9312
04-13 08:24:19.110+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:19.110+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:20.135+0900 D/sio_packet(11608): from json
04-13 08:24:20.135+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:20.135+0900 D/sio_packet(11608): from json
04-13 08:24:20.135+0900 D/sio_packet(11608): from json
04-13 08:24:20.135+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:20.135+0900 D/sio_packet(11608): from json
04-13 08:24:20.135+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:20.135+0900 D/sio_packet(11608): from json
04-13 08:24:20.135+0900 D/sio_packet(11608): IsInt64
04-13 08:24:20.135+0900 D/sio_packet(11608): from json
04-13 08:24:20.135+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:20.135+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:20.135+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:20.135+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:20.135+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:20.135+0900 F/socket.io(11608): key size : 9312
04-13 08:24:20.135+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:20.135+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:20.440+0900 D/sio_packet(11608): from json
04-13 08:24:20.440+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:20.440+0900 D/sio_packet(11608): from json
04-13 08:24:20.440+0900 D/sio_packet(11608): from json
04-13 08:24:20.440+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:20.440+0900 D/sio_packet(11608): from json
04-13 08:24:20.440+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:21.165+0900 D/sio_packet(11608): from json
04-13 08:24:21.165+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:21.165+0900 D/sio_packet(11608): from json
04-13 08:24:21.165+0900 D/sio_packet(11608): from json
04-13 08:24:21.165+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:21.165+0900 D/sio_packet(11608): from json
04-13 08:24:21.165+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:21.165+0900 D/sio_packet(11608): from json
04-13 08:24:21.165+0900 D/sio_packet(11608): IsInt64
04-13 08:24:21.165+0900 D/sio_packet(11608): from json
04-13 08:24:21.165+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:21.165+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:21.165+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:21.165+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:21.165+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:21.165+0900 F/socket.io(11608): key size : 9312
04-13 08:24:21.165+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:21.165+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:22.185+0900 D/sio_packet(11608): from json
04-13 08:24:22.185+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:22.185+0900 D/sio_packet(11608): from json
04-13 08:24:22.185+0900 D/sio_packet(11608): from json
04-13 08:24:22.185+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:22.185+0900 D/sio_packet(11608): from json
04-13 08:24:22.185+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:22.185+0900 D/sio_packet(11608): from json
04-13 08:24:22.185+0900 D/sio_packet(11608): IsInt64
04-13 08:24:22.185+0900 D/sio_packet(11608): from json
04-13 08:24:22.185+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:22.185+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:22.185+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:22.185+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:22.185+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:22.185+0900 F/socket.io(11608): key size : 9312
04-13 08:24:22.185+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:22.185+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:22.490+0900 D/sio_packet(11608): from json
04-13 08:24:22.490+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:22.490+0900 D/sio_packet(11608): from json
04-13 08:24:22.490+0900 D/sio_packet(11608): from json
04-13 08:24:22.490+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:22.490+0900 D/sio_packet(11608): from json
04-13 08:24:22.490+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:23.200+0900 D/sio_packet(11608): from json
04-13 08:24:23.200+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:23.200+0900 D/sio_packet(11608): from json
04-13 08:24:23.200+0900 D/sio_packet(11608): from json
04-13 08:24:23.200+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:23.200+0900 D/sio_packet(11608): from json
04-13 08:24:23.200+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:23.200+0900 D/sio_packet(11608): from json
04-13 08:24:23.200+0900 D/sio_packet(11608): IsInt64
04-13 08:24:23.200+0900 D/sio_packet(11608): from json
04-13 08:24:23.200+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:23.200+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:23.200+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:23.200+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:23.200+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:23.200+0900 F/socket.io(11608): key size : 9312
04-13 08:24:23.200+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:23.200+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:23.375+0900 F/sio_packet(11608): accept()
04-13 08:24:24.335+0900 D/sio_packet(11608): from json
04-13 08:24:24.335+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:24.335+0900 D/sio_packet(11608): from json
04-13 08:24:24.335+0900 D/sio_packet(11608): from json
04-13 08:24:24.335+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:24.335+0900 D/sio_packet(11608): from json
04-13 08:24:24.335+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:24.335+0900 D/sio_packet(11608): from json
04-13 08:24:24.335+0900 D/sio_packet(11608): IsInt64
04-13 08:24:24.335+0900 D/sio_packet(11608): from json
04-13 08:24:24.335+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:24.335+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:24.335+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:24.335+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:24.335+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:24.335+0900 F/socket.io(11608): key size : 9312
04-13 08:24:24.335+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:24.335+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:24.390+0900 D/sio_packet(11608): from json
04-13 08:24:24.390+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:24.390+0900 D/sio_packet(11608): from json
04-13 08:24:24.390+0900 D/sio_packet(11608): from json
04-13 08:24:24.390+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:24.390+0900 D/sio_packet(11608): from json
04-13 08:24:24.390+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:25.250+0900 D/sio_packet(11608): from json
04-13 08:24:25.250+0900 D/value.IsArray()(11608): if arr from json
04-13 08:24:25.250+0900 D/sio_packet(11608): from json
04-13 08:24:25.250+0900 D/sio_packet(11608): from json
04-13 08:24:25.250+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:25.250+0900 D/sio_packet(11608): from json
04-13 08:24:25.250+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:25.250+0900 D/sio_packet(11608): from json
04-13 08:24:25.250+0900 D/sio_packet(11608): IsInt64
04-13 08:24:25.250+0900 D/sio_packet(11608): from json
04-13 08:24:25.250+0900 D/value.IsObject()(11608): if binary from json
04-13 08:24:25.250+0900 F/value.IsObject()(11608): num!!!!!0
04-13 08:24:25.250+0900 F/value.IsObject()(11608): buf size!!!!!1
04-13 08:24:25.250+0900 E/socket.io(11608): 669: Received Message type(Event)
04-13 08:24:25.250+0900 F/socket.io(11608): bind_event [test2] 11608
04-13 08:24:25.250+0900 F/socket.io(11608): key size : 9312
04-13 08:24:25.250+0900 F/get_binary(11608): in get binary_message()...
04-13 08:24:25.250+0900 F/socket.io(11608): texture_getter value -1276401640
04-13 08:24:25.295+0900 D/EFL     (11608): ecore_x<11608> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=277534105 button=1
04-13 08:24:25.360+0900 D/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600044  register trigger_timer!  pointed_win=0x653ab1 
04-13 08:24:25.360+0900 D/EFL     (11608): ecore_x<11608> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=277534177 button=1
04-13 08:24:25.380+0900 F/socket.io(11608): !texture_getter value -1276401640
04-13 08:24:25.700+0900 W/CRASH_MANAGER(11621): worker.c: worker_job(1189) > 11116086f7468142888106
