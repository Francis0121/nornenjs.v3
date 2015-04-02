S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: hyok
PID: 10360
Date: 2015-03-31 16:17:24+0900
Executable File Path: /opt/usr/apps/org.tizen.hyok/bin/hyok
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0x47

Register Information
r0   = 0xb70d60a9, r1   = 0x00000047
r2   = 0xb657a2cc, r3   = 0x0000003c
r4   = 0xb70bd4c0, r5   = 0xb70c6640
r6   = 0xb657a2cc, r7   = 0xb70d60a8
r8   = 0x00000047, r9   = 0x00000001
r10  = 0xb70ca3d0, fp   = 0xb657a2cc
ip   = 0xb62bc754, sp   = 0xbeb96f50
lr   = 0xb62a42d5, pc   = 0xb6d55968
cpsr = 0x00000010

Memory Information
MemTotal:   797840 KB
MemFree:    418796 KB
Buffers:     29476 KB
Cached:     129572 KB
VmPeak:      66460 KB
VmSize:      66456 KB
VmLck:           0 KB
VmHWM:       12360 KB
VmRSS:       12360 KB
VmData:      12104 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       23612 KB
VmPTE:          42 KB
VmSwap:          0 KB

Threads Information
Threads: 2
PID = 10360 TID = 10360
10360 10364 

Maps Information
b31f6000 b31f7000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b31ff000 b3206000 r-xp /usr/lib/libfeedback.so.0.1.4
b3219000 b321a000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b3222000 b3239000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b3374000 b3379000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b3382000 b338c000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3395000 b33a0000 r-xp /usr/lib/evas/modules/engines/software_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3ba9000 b3baf000 r-xp /usr/lib/libUMP.so
b3bb7000 b3bbe000 r-xp /usr/lib/libtbm.so.1.0.0
b3bc6000 b3bcb000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3bd3000 b3bda000 r-xp /usr/lib/libdrm.so.2.4.0
b3be3000 b3be5000 r-xp /usr/lib/libdri2.so.0.0.0
b3bed000 b3c00000 r-xp /usr/lib/libEGL_platform.so
b3c09000 b3ce0000 r-xp /usr/lib/libMali.so
b3ceb000 b3cf2000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3cfa000 b3cff000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3d07000 b3d1e000 r-xp /usr/lib/libEGL.so.1.4
b3d27000 b3d2c000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3d35000 b3d36000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3d3f000 b3d57000 r-xp /usr/lib/libpng12.so.0.50.0
b3d5f000 b3d9d000 r-xp /usr/lib/libGLESv2.so.2.0
b3da5000 b3da9000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3db2000 b3db5000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3dbe000 b3e74000 r-xp /usr/lib/libcairo.so.2.11200.14
b3e7f000 b3e95000 r-xp /usr/lib/libtts.so
b3e9e000 b3eaf000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3eb7000 b3eb9000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3ec1000 b3ec9000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3ed1000 b3edb000 r-xp /lib/libnss_files-2.13.so
b3eea000 b3eec000 r-xp /opt/usr/apps/org.tizen.hyok/bin/hyok
b40f4000 b4115000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b411e000 b413b000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b4144000 b4212000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4229000 b424f000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4259000 b425b000 r-xp /usr/lib/libiniparser.so.0
b4265000 b426b000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b4274000 b427a000 r-xp /usr/lib/libappsvc.so.0.1.0
b4283000 b4285000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b428e000 b4292000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b429a000 b429e000 r-xp /usr/lib/libogg.so.0.7.1
b42a6000 b42c8000 r-xp /usr/lib/libvorbis.so.0.4.3
b42d0000 b43b4000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b43c8000 b43f9000 r-xp /usr/lib/libFLAC.so.8.2.0
b4402000 b4404000 r-xp /usr/lib/libXau.so.6.0.0
b440c000 b4458000 r-xp /usr/lib/libssl.so.1.0.0
b4465000 b4493000 r-xp /usr/lib/libidn.so.11.5.44
b449b000 b44a5000 r-xp /usr/lib/libcares.so.2.1.0
b44ad000 b44f2000 r-xp /usr/lib/libsndfile.so.1.0.25
b4500000 b4507000 r-xp /usr/lib/libsensord-share.so
b450f000 b4525000 r-xp /lib/libexpat.so.1.5.2
b4533000 b4536000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b453e000 b4572000 r-xp /usr/lib/libicule.so.51.1
b457b000 b458e000 r-xp /usr/lib/libxcb.so.1.1.0
b4596000 b45d1000 r-xp /usr/lib/libcurl.so.4.3.0
b45da000 b45e3000 r-xp /usr/lib/libethumb.so.1.7.99
b5b51000 b5be5000 r-xp /usr/lib/libstdc++.so.6.0.16
b5bf8000 b5bfa000 r-xp /usr/lib/libctxdata.so.0.0.0
b5c02000 b5c0f000 r-xp /usr/lib/libremix.so.0.0.0
b5c17000 b5c18000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c20000 b5c37000 r-xp /usr/lib/liblua-5.1.so
b5c40000 b5c47000 r-xp /usr/lib/libembryo.so.1.7.99
b5c4f000 b5c72000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c8a000 b5ca0000 r-xp /usr/lib/libsensor.so.1.1.0
b5ca9000 b5cff000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5d0c000 b5d2f000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d38000 b5d7e000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d87000 b5d9a000 r-xp /usr/lib/libfribidi.so.0.3.1
b5da2000 b5df2000 r-xp /usr/lib/libfreetype.so.6.8.1
b5dfd000 b5e00000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5e08000 b5e0c000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e14000 b5e19000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e22000 b5e2c000 r-xp /usr/lib/libXext.so.6.4.0
b5e34000 b5f15000 r-xp /usr/lib/libX11.so.6.3.0
b5f20000 b5f23000 r-xp /usr/lib/libXtst.so.6.1.0
b5f2b000 b5f31000 r-xp /usr/lib/libXrender.so.1.3.0
b5f39000 b5f3e000 r-xp /usr/lib/libXrandr.so.2.2.0
b5f46000 b5f47000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f50000 b5f58000 r-xp /usr/lib/libXi.so.6.1.0
b5f59000 b5f5c000 r-xp /usr/lib/libXfixes.so.3.1.0
b5f64000 b5f66000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f6e000 b5f70000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f78000 b5f79000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f82000 b5f88000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f91000 b5faa000 r-xp /usr/lib/libecore_con.so.1.7.99
b5fb4000 b5fba000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5fc2000 b5fca000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5fd2000 b5fd6000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5fdf000 b5ff5000 r-xp /usr/lib/libefreet.so.1.7.99
b5ffe000 b6007000 r-xp /usr/lib/libedbus.so.1.7.99
b600f000 b60f4000 r-xp /usr/lib/libicuuc.so.51.1
b6109000 b6248000 r-xp /usr/lib/libicui18n.so.51.1
b6258000 b62b4000 r-xp /usr/lib/libedje.so.1.7.99
b62be000 b62cf000 r-xp /usr/lib/libecore_input.so.1.7.99
b62d7000 b62dc000 r-xp /usr/lib/libecore_file.so.1.7.99
b62e4000 b62fd000 r-xp /usr/lib/libeet.so.1.7.99
b630e000 b6312000 r-xp /usr/lib/libappcore-common.so.1.1
b631b000 b63e7000 r-xp /usr/lib/libevas.so.1.7.99
b640c000 b642d000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6436000 b6465000 r-xp /usr/lib/libecore_x.so.1.7.99
b646f000 b65a3000 r-xp /usr/lib/libelementary.so.1.7.99
b65bb000 b65bc000 r-xp /usr/lib/libjournal.so.0.1.0
b65c5000 b6690000 r-xp /usr/lib/libxml2.so.2.7.8
b669e000 b66ae000 r-xp /lib/libresolv-2.13.so
b66b2000 b66c8000 r-xp /lib/libz.so.1.2.5
b66d0000 b66d2000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b66da000 b66df000 r-xp /usr/lib/libffi.so.5.0.10
b66e8000 b66e9000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b66f1000 b66f4000 r-xp /lib/libattr.so.1.1.0
b66fc000 b68a4000 r-xp /usr/lib/libcrypto.so.1.0.0
b68c4000 b68de000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b68e7000 b6950000 r-xp /lib/libm-2.13.so
b6959000 b6999000 r-xp /usr/lib/libeina.so.1.7.99
b69a2000 b69aa000 r-xp /usr/lib/libvconf.so.0.2.45
b69b2000 b69b5000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b69bd000 b69f1000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b69fa000 b6ace000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6ada000 b6ae0000 r-xp /lib/librt-2.13.so
b6ae9000 b6aee000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6af7000 b6afe000 r-xp /lib/libcrypt-2.13.so
b6b2e000 b6b31000 r-xp /lib/libcap.so.2.21
b6b39000 b6b3b000 r-xp /usr/lib/libiri.so
b6b43000 b6b62000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b6a000 b6b80000 r-xp /usr/lib/libecore.so.1.7.99
b6b96000 b6b9b000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6ba4000 b6c74000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c75000 b6c83000 r-xp /usr/lib/libail.so.0.1.0
b6c8b000 b6ca2000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6cab000 b6cb5000 r-xp /lib/libunwind.so.8.0.1
b6ce3000 b6dfe000 r-xp /lib/libc-2.13.so
b6e0c000 b6e14000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e1c000 b6e46000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e4f000 b6e52000 r-xp /usr/lib/libbundle.so.0.1.22
b6e5a000 b6e5c000 r-xp /lib/libdl-2.13.so
b6e65000 b6e68000 r-xp /usr/lib/libsmack.so.1.0.0
b6e70000 b6ed2000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6edc000 b6eee000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6ef7000 b6f0b000 r-xp /lib/libpthread-2.13.so
b6f18000 b6f1c000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f26000 b6f28000 r-xp /usr/lib/libdlog.so.0.0.0
b6f30000 b6f3b000 r-xp /usr/lib/libaul.so.0.1.0
b6f45000 b6f49000 r-xp /usr/lib/libsys-assert.so
b6f52000 b6f6f000 r-xp /lib/ld-2.13.so
b6f78000 b6f7e000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f86000 b6fb2000 rw-p [heap]
b6fb2000 b70f3000 rw-p [heap]
beb77000 beb98000 rwxp [stack]
End of Maps Information

Callstack Information (PID:10360)
Call Stack Count: 15
 0: strcmp + 0x4 (0xb6d55968) [/lib/libc.so.6] + 0x72968
 1: (0xb62a42d5) [/usr/lib/libedje.so.1] + 0x4c2d5
 2: edje_object_part_text_escaped_set + 0xca (0xb62a908f) [/usr/lib/libedje.so.1] + 0x5108f
 3: (0xb651dcfd) [/usr/lib/libelementary.so.1] + 0xaecfd
 4: (0xb6519fad) [/usr/lib/libelementary.so.1] + 0xaafad
 5: elm_layout_text_set + 0x4a (0xb651d497) [/usr/lib/libelementary.so.1] + 0xae497
 6: create_base_gui + 0x110 (0xb3eeae55) [/opt/usr/apps/org.tizen.hyok/bin/hyok] + 0xe55
 7: app_create + 0x12 (0xb3eeaed3) [/opt/usr/apps/org.tizen.hyok/bin/hyok] + 0xed3
 8: (0xb45342ff) [/usr/lib/libcapi-appfw-application.so.0] + 0x12ff
 9: appcore_efl_main + 0x1de (0xb6f1a62f) [/usr/lib/libappcore-efl.so.1] + 0x262f
10: ui_app_main + 0xb0 (0xb4534c79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
11: main + 0x152 (0xb3eeb10b) [/opt/usr/apps/org.tizen.hyok/bin/hyok] + 0x110b
12: (0xb6f7ac6f) [/opt/usr/apps/org.tizen.hyok/bin/hyok] + 0x2c6f
13: __libc_start_main + 0x114 (0xb6cfa82c) [/lib/libc.so.6] + 0x1782c
14: (0xb6f7b35c) [/opt/usr/apps/org.tizen.hyok/bin/hyok] + 0x335c
End of Call Stack

Package Information
Package Name: org.tizen.hyok
Package ID : org.tizen.hyok
Version: 1.0.0
Package Type: coretpk
App Name: hyok
App ID: org.tizen.hyok
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
 indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
03-31 16:16:31.745+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
03-31 16:16:31.745+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.745+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 3"
03-31 16:16:31.745+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1211) > handle_more_notify_icon[1211]	 "LANDSCAPE :: 3"
03-31 16:16:31.745+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1220) > handle_more_notify_icon[1220]	 "LANDSCAPE :: handle_more_notify_hide"
03-31 16:16:31.745+0900 D/indicator( 2226): active_sync.c: wake_up_cb(277) > wake_up_cb[277]	 "wake up 0"
03-31 16:16:31.745+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(199) > indicator_active_sync_change_cb[199]	 "Error getting VCONFKEY_EAS_SYNC_STATE value"
03-31 16:16:31.745+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(213) > indicator_active_sync_change_cb[213]	 "Error getting VCONFKEY_FACEBOOK_SYNC_STATE value"
03-31 16:16:31.750+0900 D/LOCKSCREEN( 2243): contextual-info.c: contextual_info_notification_page_hide(865) > [contextual_info_notification_page_hide:865 ] Notification page doesn't exist
03-31 16:16:31.750+0900 W/LOCKSCREEN( 2243): daemon.c: _state_transit(390) > [_state_transit:390:W] state transit:4
03-31 16:16:31.755+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(220) > indicator_active_sync_change_cb[220]	 "Error getting VCONFKEY_GOOGLE_PIM_SYNC_STATE value"
03-31 16:16:31.755+0900 E/LOCKSCREEN( 2243): daemon.c: _is_pwlock_enabled(231) > [_is_pwlock_enabled:231:E] failed to get pwlock status
03-31 16:16:31.755+0900 E/LOCKSCREEN( 2243): daemon.c: _is_testmode_enabled(214) > [_is_testmode_enabled:214:E] failed to get testmode status
03-31 16:16:31.755+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(227) > indicator_active_sync_change_cb[227]	 "Error getting VCONFKEY_CLOUD_PDM_SYNC_STATE value"
03-31 16:16:31.755+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(235) > indicator_active_sync_change_cb[235]	 "Error getting VCONFKEY_OMA_DS_SYNC_STATE value"
03-31 16:16:31.755+0900 E/LOCKSCREEN( 2243): util-daemon.c: lockd_call_state_get(159) > [lockd_call_state_get:159:E] Failed to get value of VCONFKEY_CALL_STATE
03-31 16:16:31.755+0900 D/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(244) > indicator_active_sync_change_cb[244]	 "Active sync is unset"
03-31 16:16:31.755+0900 W/LOCKSCREEN( 2243): daemon.c: _state_enter(329) > [_state_enter:329:W] WILL RESUME => RESUME
03-31 16:16:31.755+0900 D/indicator( 2226): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
03-31 16:16:31.755+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
03-31 16:16:31.755+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
03-31 16:16:31.755+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 16:16 5 HH:mm"
03-31 16:16:31.755+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 16:16"
03-31 16:16:31.755+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 16&#x2236;16"
03-31 16:16:31.760+0900 D/indicator( 2226): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 366864 Time: <font_size=34>16&#x2236;16</font_size></font>"
03-31 16:16:31.760+0900 D/indicator( 2226): battery.c: indicator_battery_update_display(862) > indicator_battery_update_display[862]	 "Battery Capacity: 100"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.battery.percentage.full.show"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.battery.percentage.full.show"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.battery.percentage.full.show"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.battery.percentage.full.show"
03-31 16:16:31.760+0900 D/indicator( 2226): battery.c: hide_digits(644) > hide_digits[644]	 "Hide digits"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 3 , MiniCtrl count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 3 , MiniCtrl count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 3 , System count : 0 , Minictrl ret : 3"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 3 , System count : 0 , Minictrl ret : 3"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
03-31 16:16:31.760+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 3"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1197) > handle_more_notify_icon[1197]	 "PORT :: 3"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1206) > handle_more_notify_icon[1206]	 "PORT :: handle_more_notify_hide"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 3 , MiniCtrl count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 3 , MiniCtrl count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 3 , System count : 0 , Minictrl ret : 3"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 3 , System count : 0 , Minictrl ret : 3"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 3"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1211) > handle_more_notify_icon[1211]	 "LANDSCAPE :: 3"
03-31 16:16:31.765+0900 D/indicator( 2226): indicator_icon_util.c: handle_more_notify_icon(1220) > handle_more_notify_icon[1220]	 "LANDSCAPE :: handle_more_notify_hide"
03-31 16:16:31.770+0900 D/indicator( 2226): active_sync.c: wake_up_cb(277) > wake_up_cb[277]	 "wake up 0"
03-31 16:16:31.770+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(199) > indicator_active_sync_change_cb[199]	 "Error getting VCONFKEY_EAS_SYNC_STATE value"
03-31 16:16:31.770+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(213) > indicator_active_sync_change_cb[213]	 "Error getting VCONFKEY_FACEBOOK_SYNC_STATE value"
03-31 16:16:31.770+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(220) > indicator_active_sync_change_cb[220]	 "Error getting VCONFKEY_GOOGLE_PIM_SYNC_STATE value"
03-31 16:16:31.770+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(227) > indicator_active_sync_change_cb[227]	 "Error getting VCONFKEY_CLOUD_PDM_SYNC_STATE value"
03-31 16:16:31.770+0900 E/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(235) > indicator_active_sync_change_cb[235]	 "Error getting VCONFKEY_OMA_DS_SYNC_STATE value"
03-31 16:16:31.770+0900 D/indicator( 2226): active_sync.c: indicator_active_sync_change_cb(244) > indicator_active_sync_change_cb[244]	 "Active sync is unset"
03-31 16:16:32.270+0900 D/DATA_PROVIDER_MASTER( 2281): client_life.c: client_is_all_paused(437) > [SECURE_LOG] 0, 0
03-31 16:16:32.270+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_vconf.c: net_nfc_server_vconf_pm_state_changed(134) > [0;35mpm_state : 1[0;m
03-31 16:16:32.270+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_vconf.c: net_nfc_server_vconf_pm_state_changed(141) > [0;35mlock_screen_set : 1[0;m
03-31 16:16:32.275+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_common.c: restart_polling_loop_thread_func(429) > [0;35mlock_screen_set:1 ,pm_state:1,lock_state:1[0;m
03-31 16:16:32.545+0900 D/EFL     ( 2243): ecore_x<2243> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=63841799 button=1
03-31 16:16:32.545+0900 D/LOCKSCREEN( 2243): progress_circle.c: unlock_mouse_down(176) > [unlock_mouse_down:176 ] 
03-31 16:16:32.545+0900 D/LOCKSCREEN( 2243): progress_circle.c: unlock_mouse_down(183) > [unlock_mouse_down:183 ] 
03-31 16:16:32.545+0900 D/LOCKSCREEN( 2243): progress_circle.c: unlock_mouse_down(192) > [unlock_mouse_down:192 ] 0x5acda8 camera_icon_x(565) camera_y(1123)
03-31 16:16:32.545+0900 D/LOCKSCREEN( 2243): progress_circle.c: unlock_mouse_down(199) > [unlock_mouse_down:199 ] touch(1023) y(60)
03-31 16:16:32.545+0900 D/LOCKSCREEN( 2243): progress_circle.c: unlock_mouse_down(261) > [unlock_mouse_down:261 ] 
03-31 16:16:32.640+0900 D/EFL     ( 2243): ecore_x<2243> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=63841892 button=1
03-31 16:16:32.640+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x600069 
03-31 16:16:32.640+0900 E/LOCKSCREEN( 2243): progress_circle.c: unlock_mouse_up(292) > 
03-31 16:16:32.640+0900 W/LOCKSCREEN( 2243): daemon.c: lockd_event(902) > [lockd_event:902:W] event:80001:VIEW_GESTURED
03-31 16:16:32.640+0900 W/LOCKSCREEN( 2243): daemon.c: _event_route(721) > [_event_route:721:W] event:80001 event_info:0
03-31 16:16:32.640+0900 W/LOCKSCREEN( 2243): view-mgr.c: _event_route(107) > [_event_route:107:W] event:80001 event_info:0
03-31 16:16:32.640+0900 W/LOCKSCREEN( 2243): view-mgr.c: _state_transit(44) > [_state_transit:44:W] state transit:2
03-31 16:16:32.640+0900 W/LOCKSCREEN( 2243): view-mgr.c: _state_transit(47) > [_state_transit:47:W] already in same state:2
03-31 16:16:32.640+0900 D/LOCKSCREEN( 2243): progress_circle.c: vi_end_signal_emit(521) > [vi_end_signal_emit:521 ] No notification imtem selected
03-31 16:16:32.640+0900 W/LOCKSCREEN( 2243): daemon.c: lockd_event_delay(915) > [lockd_event_delay:915:W] dealyed event:2:UNLOCK wait for:0.500000
03-31 16:16:33.155+0900 W/LOCKSCREEN( 2243): daemon.c: _event_route(721) > [_event_route:721:W] event:2 event_info:0
03-31 16:16:33.155+0900 W/LOCKSCREEN( 2243): daemon.c: _state_transit(390) > [_state_transit:390:W] state transit:6
03-31 16:16:33.155+0900 E/LOCKSCREEN( 2243): daemon.c: _is_pwlock_enabled(231) > [_is_pwlock_enabled:231:E] failed to get pwlock status
03-31 16:16:33.155+0900 E/LOCKSCREEN( 2243): daemon.c: _is_testmode_enabled(214) > [_is_testmode_enabled:214:E] failed to get testmode status
03-31 16:16:33.155+0900 E/LOCKSCREEN( 2243): util-daemon.c: lockd_call_state_get(159) > [lockd_call_state_get:159:E] Failed to get value of VCONFKEY_CALL_STATE
03-31 16:16:33.155+0900 W/LOCKSCREEN( 2243): daemon.c: _state_enter(329) > [_state_enter:329:W] RESUME => HIDE
03-31 16:16:33.160+0900 D/LOCKSCREEN( 2243): complex-password.c: _pause(698) > [_pause:698 ] 
03-31 16:16:33.160+0900 E/LOCKSCREEN( 2243): weather.c: _event_deregister(370) > [_event_deregister:370:E] Failed to notify VCONFKEY_DAILYBRIEFING_WEATHER_MY_LOCATION
03-31 16:16:33.160+0900 E/LOCKSCREEN( 2243): weather.c: _event_deregister(373) > [_event_deregister:373:E] Failed to notify VCONFKEY_DAILYBRIEFING_WEATHER_ICON_ID
03-31 16:16:33.160+0900 E/LOCKSCREEN( 2243): weather.c: _event_deregister(376) > [_event_deregister:376:E] Failed to notify VCONFKEY_DAILYBRIEFING_WEATHER_TEMP
03-31 16:16:33.165+0900 E/LOCKSCREEN( 2243): weather.c: _event_deregister(379) > [_event_deregister:379:E] Failed to notify VCONFKEY_DAILYBRIEFING_WEATHER_TEMP_UNIT
03-31 16:16:33.165+0900 E/LOCKSCREEN( 2243): weather.c: _event_deregister(382) > [_event_deregister:382:E] Failed to notify VCONFKEY_DAILYBRIEFING_WEATHER_CITY_NAME
03-31 16:16:33.170+0900 W/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2252
03-31 16:16:33.170+0900 D/indicator( 2226): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1a00008"
03-31 16:16:33.170+0900 D/indicator( 2226): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1c00003"
03-31 16:16:33.170+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
03-31 16:16:33.170+0900 D/indicator( 2226): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1c00003"
03-31 16:16:33.170+0900 D/indicator( 2226): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
03-31 16:16:33.175+0900 D/APP_CORE( 2252): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 0
03-31 16:16:33.175+0900 D/APP_CORE( 2252): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active 0
03-31 16:16:33.175+0900 D/APP_CORE( 2252): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
03-31 16:16:33.180+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(470) > [APP 2252] Event: RESUME State: PAUSED
03-31 16:16:33.180+0900 D/LAUNCH  ( 2252): appcore-efl.c: __do_app(557) > [menu-screen:Application:resume:start]
03-31 16:16:33.180+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(559) > [APP 2252] RESUME
03-31 16:16:33.180+0900 I/CAPI_APPFW_APPLICATION( 2252): app_main.c: app_appcore_resume(216) > app_appcore_resume
03-31 16:16:33.180+0900 D/MENU_SCREEN( 2252): menu_screen.c: _resume_cb(547) > START RESUME
03-31 16:16:33.180+0900 D/MENU_SCREEN( 2252): page_scroller.c: page_scroller_focus(1133) > Focus set scroller(0xb71a7398), page:0, item:cube
03-31 16:16:33.180+0900 D/LAUNCH  ( 2252): appcore-efl.c: __do_app(567) > [menu-screen:Application:resume:done]
03-31 16:16:33.180+0900 D/LAUNCH  ( 2252): appcore-efl.c: __do_app(569) > [menu-screen:Application:Launching:done]
03-31 16:16:33.180+0900 D/APP_CORE( 2252): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
03-31 16:16:33.180+0900 E/APP_CORE( 2252): appcore-efl.c: __trm_app_info_send_socket(233) > access
03-31 16:16:33.180+0900 W/LOCKSCREEN( 2243): lockscreen-lite.c: _window_visibility_cb(294) > [_window_visibility_cb:294:W] Window [0x1A00008] is now visible(1)
03-31 16:16:33.180+0900 W/LOCKSCREEN( 2243): daemon.c: lockd_event(902) > [lockd_event:902:W] event:10002:WIN_BECOME_INVISIBLE
03-31 16:16:33.180+0900 W/LOCKSCREEN( 2243): daemon.c: _event_route(721) > [_event_route:721:W] event:10002 event_info:0
03-31 16:16:33.180+0900 D/STARTER ( 2230): lock-daemon-lite.c: _lockd_notify_lock_state_cb(914) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:914:D] lock state changed!!
03-31 16:16:33.180+0900 D/STARTER ( 2230): lock-daemon-lite.c: lockd_get_lock_state(366) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:366:D] Get lock state : 0
03-31 16:16:33.180+0900 D/STARTER ( 2230): lock-daemon-lite.c: _lockd_notify_lock_state_cb(929) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:929:D] unlocked..!!
03-31 16:16:33.180+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_common.c: restart_polling_loop_thread_func(429) > [0;35mlock_screen_set:1 ,pm_state:1,lock_state:0[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: net_nfc_nxp_controller_configure_discovery(2133) > [0;36mrun poll loop[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2038) > [0;36mset polling loop configure = [0xffffffff][0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2045) > [0;36mturn on ISO14443A[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2056) > [0;36mturn on ISO14443B[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2067) > [0;36mturn on Felica[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2080) > [0;36mturn on ISO15693[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2091) > [0;36mturn on NFCIP[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_Mgt_ConfigureDiscovery(155) > [0;34m>>> Function [phLibNfc_Mgt_ConfigureDiscovery] is called Line:155 <<<[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_Mgt_ConfigureDiscovery(157) > [0;34mDiscoveryMode: 0, sADDSetup: 0xb2fdc67f, pConfigDiscovery_RspCb: 0xf4240 pContext:0xb68f0027[0;m
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_Mgt_ConfigureDiscovery(189) > [0;34m PollEnabled 3002975871 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableIso14443A 1 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableIso14443B 1 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableFelica212 1 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableNfcActive 1 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableIso15693 1 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  NfcIP_Mode 39 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  DisableCardEmulation 0 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  Duration 1000000 
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  NfcIP_Tgt_Disable 0
03-31 16:16:33.180+0900 D/NFC_PLUGIN_NXP_STACK( 2285): [0;m
03-31 16:16:33.180+0900 D/VOLUME  ( 2251): volume_control.c: _idle_lock_state_vconf_changed_cb(763) > [_idle_lock_state_vconf_changed_cb:763] idle lock state : 0
03-31 16:16:33.180+0900 D/RESOURCED( 2367): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2243, type = 2
03-31 16:16:33.185+0900 D/APP_CORE( 2243): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1a00008 fully_obscured 1
03-31 16:16:33.185+0900 D/RESOURCED( 2367): cpu.c: cpu_background_state(100) > [cpu_background_state,100] cpu_background_state : pid = 2243, appname = lockscreen
03-31 16:16:33.185+0900 D/RESOURCED( 2367): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/service/cgroup.procs, value 2243
03-31 16:16:33.185+0900 D/APP_CORE( 2243): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
03-31 16:16:33.185+0900 D/APP_CORE( 2243): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
03-31 16:16:33.185+0900 D/APP_CORE( 2243): appcore-efl.c: __do_app(470) > [APP 2243] Event: PAUSE State: PAUSED
03-31 16:16:33.185+0900 D/APP_CORE( 2243): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
03-31 16:16:33.185+0900 E/APP_CORE( 2243): appcore-efl.c: __trm_app_info_send_socket(233) > access
03-31 16:16:33.190+0900 I/RESOURCED( 2367): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/background/cgroup.procs, pid : 2243, oom : 300
03-31 16:16:33.190+0900 E/RESOURCED( 2367): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/background/cgroup.procs open failed
03-31 16:16:33.190+0900 E/RESOURCED( 2367): proc-main.c: proc_update_process_state(233) > [proc_update_process_state,233] Current pid (2243) didn't have any process list
03-31 16:16:33.205+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT clock.font.24"
03-31 16:16:33.205+0900 D/indicator( 2226): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission clock.font.24"
03-31 16:16:33.205+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_config_discovery_cb(133) > [0;34m>>> Callback function [phLibNfc_config_discovery_cb] is called Line:133 <<<[0;m
03-31 16:16:33.205+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_config_discovery_cb(134) > [0;34mstatus: 0x0[0;m
03-31 16:16:33.205+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_configure_discovery_cb(791) > [0;36mconfigure discovery is successful[0;m
03-31 16:16:33.205+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: net_nfc_nxp_controller_configure_discovery(2154) > [0;36mdiscovering config is end[0;m
03-31 16:16:33.205+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_common.c: restart_polling_loop_thread_func(491) > [0;35mpolling enable[0;m
03-31 16:16:33.315+0900 D/APP_CORE( 8451): appcore-rotation.c: __changed_cb(123) > [APP 8451] Rotation: 1 -> 4
03-31 16:16:33.315+0900 D/APP_CORE( 8451): appcore-rotation.c: __changed_cb(126) > [APP 8451] Rotation: 1 -> 4
03-31 16:16:33.315+0900 I/CAPI_APPFW_APPLICATION( 8451): app_main.c: _ui_app_appcore_rotation_event(482) > _ui_app_appcore_rotation_event
03-31 16:16:33.340+0900 D/LOCKSCREEN( 2243): util-daemon.c: lockd_cache_flush(168) > [lockd_cache_flush:168 ] 
03-31 16:16:33.340+0900 D/APP_CORE( 2243): appcore-efl.c: __do_app(470) > [APP 2243] Event: MEM_FLUSH State: PAUSED
03-31 16:16:33.340+0900 W/LOCKSCREEN( 2243): view-mgr.c: _event_route(107) > [_event_route:107:W] event:2 event_info:0
03-31 16:16:33.340+0900 D/LOCKSCREEN( 2243): default-unlock.c: default_unlock_view_back_change(248) > [default_unlock_view_back_change:248 ] 
03-31 16:16:33.340+0900 D/LOCKSCREEN( 2243): property.c: _vconf_init(236) > [_vconf_init:236 ] 
03-31 16:16:33.340+0900 E/LOCKSCREEN( 2243): property.c: _vconf_init(243) > [_vconf_init:243:E] Failed to listen for db/lockscreen/weather_display
03-31 16:16:33.340+0900 E/LOCKSCREEN( 2243): property.c: _vconf_get(311) > [_vconf_get:311:E] Failed to get value of db/lockscreen/weather_display
03-31 16:16:33.340+0900 E/LOCKSCREEN( 2243): property.c: _listen(364) > [_listen:364:E] Failed to listen 4
03-31 16:16:33.340+0900 E/LOCKSCREEN( 2243): property.c: _vconf_get(311) > [_vconf_get:311:E] Failed to get value of db/lockscreen/weather_display
03-31 16:16:33.340+0900 D/LOCKSCREEN( 2243): default-unlock.c: default_unlock_hide_on_noti_tapped(421) > Hide layout change on noti tapped
03-31 16:16:33.340+0900 E/LOCKSCREEN( 2243): default-unlock.c: default_unlock_hide_on_noti_tapped(424) > Failed to get selected noti
03-31 16:16:33.340+0900 W/LOCKSCREEN( 2243): daemon.c: lockd_event(902) > [lockd_event:902:W] event:80000:VIEW_IDLE
03-31 16:16:33.340+0900 W/LOCKSCREEN( 2243): daemon.c: _event_route(721) > [_event_route:721:W] event:80000 event_info:0
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): control-panel.c: _lockscreen_set_bg(698) > [_lockscreen_set_bg:698 ] org.tizen.system.deviced.display-LockScreenBgOn : true(1) 
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): lockscreen-lite.c: destroy_password_layout(973) > [destroy_password_layout:973 ] PASSWORD_DESTROYED
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): control-panel-password.c: control_panel_password_destroy(371) > [control_panel_password_destroy:371 ] 
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): control-panel.c: destroy_emg_button(381) > [destroy_emg_button:381 ] 
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): control-panel.c: control_panel_sim_state_changed(264) > [control_panel_sim_state_changed:264 ] 
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): sim-state.c: sim_status_network_name_get(223) > 
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): sim-state.c: sim_status_network_name_get(232) > [sim_status_network_name_get:232 ] Sim card slots are empty
03-31 16:16:33.350+0900 D/LOCKSCREEN( 2243): control-panel.c: control_panel_sim_state_changed(294) > [control_panel_sim_state_changed:294 ] Final network name : [Emergency calls only]
03-31 16:16:33.355+0900 E/LOCKSCREEN( 2243): background-view.c: background_image_next_set(178) > [background_image_next_set:178:E] fopen wallpaper txt file failed.
03-31 16:16:34.020+0900 W/LOCKSCREEN( 2243): util-daemon.c: _ckmc_unlock_timer_cb(195) > [_ckmc_unlock_timer_cb:195:W] key manager unlock:0
03-31 16:16:56.065+0900 D/RESOURCED( 2367): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
03-31 16:16:56.355+0900 D/DATA_PROVIDER_MASTER( 2281): client_life.c: client_is_all_paused(437) > [SECURE_LOG] 0, 0
03-31 16:16:56.355+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_vconf.c: net_nfc_server_vconf_pm_state_changed(134) > [0;35mpm_state : 2[0;m
03-31 16:16:56.355+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_vconf.c: net_nfc_server_vconf_pm_state_changed(141) > [0;35mlock_screen_set : 1[0;m
03-31 16:16:57.540+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=63866796 button=1
03-31 16:16:57.545+0900 D/MENU_SCREEN( 2252): mouse.c: _down_cb(103) > Mouse down (357,765)
03-31 16:16:57.545+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_vconf.c: net_nfc_server_vconf_pm_state_changed(134) > [0;35mpm_state : 1[0;m
03-31 16:16:57.545+0900 D/DATA_PROVIDER_MASTER( 2281): client_life.c: client_is_all_paused(437) > [SECURE_LOG] 0, 0
03-31 16:16:57.545+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_vconf.c: net_nfc_server_vconf_pm_state_changed(141) > [0;35mlock_screen_set : 1[0;m
03-31 16:16:57.545+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_common.c: restart_polling_loop_thread_func(429) > [0;35mlock_screen_set:1 ,pm_state:1,lock_state:0[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: net_nfc_nxp_controller_configure_discovery(2133) > [0;36mrun poll loop[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2038) > [0;36mset polling loop configure = [0xffffffff][0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2045) > [0;36mturn on ISO14443A[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2056) > [0;36mturn on ISO14443B[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2067) > [0;36mturn on Felica[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2080) > [0;36mturn on ISO15693[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_nxp_set_add_config(2091) > [0;36mturn on NFCIP[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_Mgt_ConfigureDiscovery(155) > [0;34m>>> Function [phLibNfc_Mgt_ConfigureDiscovery] is called Line:155 <<<[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_Mgt_ConfigureDiscovery(157) > [0;34mDiscoveryMode: 0, sADDSetup: 0xb2fdc67f, pConfigDiscovery_RspCb: 0xf4240 pContext:0xb68f0027[0;m
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_Mgt_ConfigureDiscovery(189) > [0;34m PollEnabled 3002975871 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableIso14443A 1 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableIso14443B 1 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableFelica212 1 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableNfcActive 1 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  EnableIso15693 1 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  NfcIP_Mode 39 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  DisableCardEmulation 0 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  Duration 1000000 
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285):  NfcIP_Tgt_Disable 0
03-31 16:16:57.545+0900 D/NFC_PLUGIN_NXP_STACK( 2285): [0;m
03-31 16:16:57.570+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_config_discovery_cb(133) > [0;34m>>> Callback function [phLibNfc_config_discovery_cb] is called Line:133 <<<[0;m
03-31 16:16:57.570+0900 D/NFC_PLUGIN_NXP_STACK( 2285): phLibNfc_discovery.c: phLibNfc_config_discovery_cb(134) > [0;34mstatus: 0x0[0;m
03-31 16:16:57.570+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: _net_nfc_configure_discovery_cb(791) > [0;36mconfigure discovery is successful[0;m
03-31 16:16:57.570+0900 D/NFC_PLUGIN_NXP_MANAGER( 2285): oem_nxp.c: net_nfc_nxp_controller_configure_discovery(2154) > [0;36mdiscovering config is end[0;m
03-31 16:16:57.570+0900 D/NET_NFC_MANAGER( 2285): net_nfc_server_common.c: restart_polling_loop_thread_func(491) > [0;35mpolling enable[0;m
03-31 16:16:58.135+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=63867386 button=1
03-31 16:16:58.135+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x60005a 
03-31 16:16:58.135+0900 D/MENU_SCREEN( 2252): mouse.c: _up_cb(120) > Mouse up (365,763)
03-31 16:16:58.540+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=63867797 button=1
03-31 16:16:58.540+0900 D/MENU_SCREEN( 2252): mouse.c: _down_cb(103) > Mouse down (389,837)
03-31 16:16:58.990+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=63868241 button=1
03-31 16:16:58.990+0900 D/MENU_SCREEN( 2252): mouse.c: _up_cb(120) > Mouse up (379,848)
03-31 16:16:58.990+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x60005a 
03-31 16:16:59.995+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_begin_handler(406) > [PROCESSMGR] ecore_x_netwm_ping_send to the client_win=0x1c00003
03-31 16:17:00.760+0900 D/indicator( 2226): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
03-31 16:17:00.765+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
03-31 16:17:00.765+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
03-31 16:17:00.770+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 16:17 5 HH:mm"
03-31 16:17:00.770+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 16:17"
03-31 16:17:00.770+0900 D/indicator( 2226): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 16&#x2236;17"
03-31 16:17:00.770+0900 D/indicator( 2226): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 367120 Time: <font_size=34>16&#x2236;17</font_size></font>"
03-31 16:17:05.000+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_confirm_handler(336) > [PROCESSMGR] last_pointed_win=0x60005a bd->visible=1
03-31 16:17:21.325+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=63890579 button=1
03-31 16:17:21.325+0900 D/MENU_SCREEN( 2252): mouse.c: _down_cb(103) > Mouse down (286,359)
03-31 16:17:22.755+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=63892014 button=1
03-31 16:17:22.760+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x60005a 
03-31 16:17:22.760+0900 D/MENU_SCREEN( 2252): mouse.c: _up_cb(120) > Mouse up (297,485)
03-31 16:17:22.945+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=63892200 button=1
03-31 16:17:22.945+0900 D/MENU_SCREEN( 2252): mouse.c: _down_cb(103) > Mouse down (245,308)
03-31 16:17:22.945+0900 D/MENU_SCREEN( 2252): item_event.c: _item_down_cb(63) > ITEM: mouse down event callback is invoked for 0xb1642190
03-31 16:17:23.765+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_begin_handler(406) > [PROCESSMGR] ecore_x_netwm_ping_send to the client_win=0x1c00003
03-31 16:17:23.985+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=63893244 button=1
03-31 16:17:23.990+0900 D/PROCESSMGR( 2097): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x60005a 
03-31 16:17:23.990+0900 D/MENU_SCREEN( 2252): mouse.c: _up_cb(120) > Mouse up (250,306)
03-31 16:17:23.990+0900 D/MENU_SCREEN( 2252): item.c: _focus_clicked_cb(668) > ITEM: mouse up event callback is invoked for 0xb1642190
03-31 16:17:23.990+0900 D/MENU_SCREEN( 2252): layout.c: layout_enable_block(116) > Enable layout blocker
03-31 16:17:23.990+0900 D/AUL     ( 2252): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.hyok
03-31 16:17:23.990+0900 D/AUL     ( 2252): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(1)
03-31 16:17:23.990+0900 D/AUL_AMD ( 2174): amd_request.c: __request_handler(491) > __request_handler: 1
03-31 16:17:23.990+0900 D/AUL_AMD ( 2174): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.hyok
03-31 16:17:23.990+0900 D/AUL_AMD ( 2174): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : org.tizen.menu-screen
03-31 16:17:24.000+0900 D/AUL_AMD ( 2174): amd_launch.c: _start_app(1591) > win(c00002) ecore_x_pointer_grab(1)
03-31 16:17:24.000+0900 E/AUL_AMD ( 2174): amd_launch.c: invoke_dbus_method_sync(1177) > dbus_connection_send error(org.freedesktop.DBus.Error.ServiceUnknown:The name org.tizen.system.coord was not provided by any .service files)
03-31 16:17:24.000+0900 D/AUL_AMD ( 2174): amd_launch.c: _start_app(1675) > org.tizen.system.coord.rotation-Degree : -74
03-31 16:17:24.000+0900 D/AUL_AMD ( 2174): amd_launch.c: _start_app(1770) > process_pool: false
03-31 16:17:24.000+0900 D/AUL_AMD ( 2174): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
03-31 16:17:24.000+0900 D/AUL_AMD ( 2174): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.hyok
03-31 16:17:24.000+0900 D/AUL_AMD ( 2174): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
03-31 16:17:24.000+0900 D/AUL     ( 2174): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(1)
03-31 16:17:24.000+0900 D/AUL_PAD ( 2214): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.hyok
03-31 16:17:24.000+0900 D/AUL_PAD ( 2214): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
03-31 16:17:24.005+0900 D/AUL_PAD ( 2214): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 10360 /opt/usr/apps/org.tizen.hyok/bin/hyok
03-31 16:17:24.005+0900 D/AUL_PAD ( 2214): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
03-31 16:17:24.005+0900 D/AUL_PAD (10360): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
03-31 16:17:24.005+0900 D/AUL_PAD (10360): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
03-31 16:17:24.005+0900 D/AUL_PAD (10360): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.hyok / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.hyok/bin/hyok 
03-31 16:17:24.030+0900 D/AUL_PAD (10360): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
03-31 16:17:24.030+0900 D/AUL_PAD (10360): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.hyok/bin/hyok##
03-31 16:17:24.030+0900 D/AUL_PAD (10360): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
03-31 16:17:24.030+0900 D/AUL_PAD (10360): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
03-31 16:17:24.030+0900 D/AUL_PAD (10360): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_APPID__##
03-31 16:17:24.030+0900 D/LAUNCH  (10360): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.hyok/bin/hyok:Platform:launchpad:done]
03-31 16:17:24.065+0900 I/CAPI_APPFW_APPLICATION(10360): app_main.c: ui_app_main(697) > app_efl_main
03-31 16:17:24.065+0900 D/LAUNCH  (10360): appcore-efl.c: appcore_efl_main(1569) > [hyok:Application:main:done]
03-31 16:17:24.105+0900 D/AUL_PAD ( 2214): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
03-31 16:17:24.105+0900 D/AUL_PAD ( 2214): sigchild.h: __send_app_launch_signal(112) > send launch signal done
03-31 16:17:24.105+0900 D/AUL_PAD ( 2214): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
03-31 16:17:24.105+0900 D/AUL     ( 2174): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
03-31 16:17:24.105+0900 E/AUL     ( 2174): simple_util.c: __trm_app_info_send_socket(264) > access
03-31 16:17:24.105+0900 D/AUL_AMD ( 2174): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 10360
03-31 16:17:24.105+0900 D/AUL     ( 2252): launch.c: app_request_to_launchpad(295) > launch request result : 10360
03-31 16:17:24.105+0900 D/MENU_SCREEN( 2252): item.c: item_launch(1003) > Launch app's ret : [10360]
03-31 16:17:24.105+0900 D/LAUNCH  ( 2252): item.c: item_launch(1005) > [org.tizen.hyok:Menuscreen:launch:done]
03-31 16:17:24.105+0900 D/MENU_SCREEN( 2252): item_event.c: _item_up_cb(85) > ITEM: mouse up event callback is invoked for 0xb1642190
03-31 16:17:24.105+0900 D/AUL_AMD ( 2174): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.hyok
03-31 16:17:24.105+0900 D/RESOURCED( 2367): proc-noti.c: recv_str(87) > [recv_str,87] str is null
03-31 16:17:24.105+0900 D/RESOURCED( 2367): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2174
03-31 16:17:24.105+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.hyok, pid 10360, type 4 
03-31 16:17:24.105+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.hyok, 10360
03-31 16:17:24.105+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.hyok with pkgname
03-31 16:17:24.105+0900 D/RESOURCED( 2367): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 10360
03-31 16:17:24.110+0900 D/RESOURCED( 2367): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 10360
03-31 16:17:24.110+0900 D/RESOURCED( 2367): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
03-31 16:17:24.110+0900 D/RESOURCED( 2367): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
03-31 16:17:24.110+0900 D/RESOURCED( 2367): datausage-common.c: populate_incomplete_counter(716) > [populate_incomplete_counter,716] Counter already exists!
03-31 16:17:24.110+0900 D/RESOURCED( 2367): datausage-common.c: populate_incomplete_counter(716) > [populate_incomplete_counter,716] Counter already exists!
03-31 16:17:24.110+0900 E/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 538
03-31 16:17:24.110+0900 D/RESOURCED( 2367): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
03-31 16:17:24.130+0900 D/APP_CORE(10360): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
03-31 16:17:24.145+0900 D/AUL     (10360): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 10360 is org.tizen.hyok
03-31 16:17:24.145+0900 D/APP_CORE(10360): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.hyok/res/locale
03-31 16:17:24.145+0900 D/APP_CORE(10360): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
03-31 16:17:24.145+0900 D/AUL     (10360): app_sock.c: __create_server_sock(135) > pg path - already exists
03-31 16:17:24.145+0900 D/LAUNCH  (10360): appcore-efl.c: __before_loop(1035) > [hyok:Platform:appcore_init:done]
03-31 16:17:24.150+0900 I/CAPI_APPFW_APPLICATION(10360): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
03-31 16:17:24.565+0900 W/CRASH_MANAGER(10365): worker.c: worker_job(1189) > 111036068796f142778624
