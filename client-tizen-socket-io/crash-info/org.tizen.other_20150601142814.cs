S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 10539
Date: 2015-06-01 14:28:14+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xaf03b004

Register Information
r0   = 0xaf03b008, r1   = 0xb3dbb76f
r2   = 0x000000e4, r3   = 0x00000000
r4   = 0xb3e658e4, r5   = 0xaf03b008
r6   = 0x00000241, r7   = 0xbeefa000
r8   = 0xb6eba2e8, r9   = 0xb6eba2e4
r10  = 0x00000000, fp   = 0xb6f11f40
ip   = 0xb3e6595c, sp   = 0xbeef9fe8
lr   = 0xb3dbb76f, pc   = 0xb6ce8150
cpsr = 0xa0000010

Memory Information
MemTotal:   797840 KB
MemFree:    431616 KB
Buffers:     23132 KB
Cached:     124860 KB
VmPeak:     148924 KB
VmSize:     119464 KB
VmLck:           0 KB
VmHWM:       16424 KB
VmRSS:       16424 KB
VmData:      63664 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24920 KB
VmPTE:          82 KB
VmSwap:          0 KB

Threads Information
Threads: 6
PID = 10539 TID = 10539
10539 10544 10545 10546 10548 10549 

Maps Information
b175a000 b175b000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17a3000 b17a4000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17ac000 b17b3000 r-xp /usr/lib/libfeedback.so.0.1.4
b17c6000 b17c7000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b17cf000 b17e6000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b196c000 b1971000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31ba000 b3205000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b320e000 b3218000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3221000 b327d000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3a8a000 b3a90000 r-xp /usr/lib/libUMP.so
b3a98000 b3aab000 r-xp /usr/lib/libEGL_platform.so
b3ab4000 b3b8b000 r-xp /usr/lib/libMali.so
b3b96000 b3bad000 r-xp /usr/lib/libEGL.so.1.4
b3bb6000 b3bbb000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3bc4000 b3bc5000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3bce000 b3be6000 r-xp /usr/lib/libpng12.so.0.50.0
b3bee000 b3c2c000 r-xp /usr/lib/libGLESv2.so.2.0
b3c34000 b3c38000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c41000 b3c43000 r-xp /usr/lib/libdri2.so.0.0.0
b3c4b000 b3c52000 r-xp /usr/lib/libdrm.so.2.4.0
b3c5b000 b3d11000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d1c000 b3d32000 r-xp /usr/lib/libtts.so
b3d3b000 b3d42000 r-xp /usr/lib/libtbm.so.1.0.0
b3d4a000 b3d4f000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3d57000 b3d68000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3d70000 b3d77000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3d7f000 b3d84000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3d8c000 b3d8e000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3d96000 b3d9e000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3da6000 b3da9000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3db2000 b3e5c000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3e67000 b3e71000 r-xp /lib/libnss_files-2.13.so
b3e7f000 b3e81000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b408a000 b40ab000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b40b4000 b40d1000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b40da000 b41a8000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b41bf000 b41e5000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b41ef000 b41f1000 r-xp /usr/lib/libiniparser.so.0
b41fb000 b4201000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b420a000 b4210000 r-xp /usr/lib/libappsvc.so.0.1.0
b4219000 b421b000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4224000 b4228000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b4230000 b4234000 r-xp /usr/lib/libogg.so.0.7.1
b423c000 b425e000 r-xp /usr/lib/libvorbis.so.0.4.3
b4266000 b434a000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b435e000 b438f000 r-xp /usr/lib/libFLAC.so.8.2.0
b4398000 b439a000 r-xp /usr/lib/libXau.so.6.0.0
b43a2000 b43ee000 r-xp /usr/lib/libssl.so.1.0.0
b43fb000 b4429000 r-xp /usr/lib/libidn.so.11.5.44
b4431000 b443b000 r-xp /usr/lib/libcares.so.2.1.0
b4443000 b4488000 r-xp /usr/lib/libsndfile.so.1.0.25
b4496000 b449d000 r-xp /usr/lib/libsensord-share.so
b44a5000 b44bb000 r-xp /lib/libexpat.so.1.5.2
b44c9000 b44cc000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b44d4000 b4508000 r-xp /usr/lib/libicule.so.51.1
b4511000 b4524000 r-xp /usr/lib/libxcb.so.1.1.0
b452c000 b4567000 r-xp /usr/lib/libcurl.so.4.3.0
b4570000 b4579000 r-xp /usr/lib/libethumb.so.1.7.99
b5ae7000 b5b7b000 r-xp /usr/lib/libstdc++.so.6.0.16
b5b8e000 b5b90000 r-xp /usr/lib/libctxdata.so.0.0.0
b5b98000 b5ba5000 r-xp /usr/lib/libremix.so.0.0.0
b5bad000 b5bae000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5bb6000 b5bcd000 r-xp /usr/lib/liblua-5.1.so
b5bd6000 b5bdd000 r-xp /usr/lib/libembryo.so.1.7.99
b5be5000 b5c08000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c20000 b5c36000 r-xp /usr/lib/libsensor.so.1.1.0
b5c3f000 b5c95000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5ca2000 b5cc5000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5cce000 b5d14000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d1d000 b5d30000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d38000 b5d88000 r-xp /usr/lib/libfreetype.so.6.8.1
b5d93000 b5d96000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5d9e000 b5da2000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5daa000 b5daf000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5db8000 b5dc2000 r-xp /usr/lib/libXext.so.6.4.0
b5dca000 b5eab000 r-xp /usr/lib/libX11.so.6.3.0
b5eb6000 b5eb9000 r-xp /usr/lib/libXtst.so.6.1.0
b5ec1000 b5ec7000 r-xp /usr/lib/libXrender.so.1.3.0
b5ecf000 b5ed4000 r-xp /usr/lib/libXrandr.so.2.2.0
b5edc000 b5edd000 r-xp /usr/lib/libXinerama.so.1.0.0
b5ee6000 b5eee000 r-xp /usr/lib/libXi.so.6.1.0
b5eef000 b5ef2000 r-xp /usr/lib/libXfixes.so.3.1.0
b5efa000 b5efc000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f04000 b5f06000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f0e000 b5f0f000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f18000 b5f1e000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f27000 b5f40000 r-xp /usr/lib/libecore_con.so.1.7.99
b5f4a000 b5f50000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5f58000 b5f60000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5f68000 b5f6c000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5f75000 b5f8b000 r-xp /usr/lib/libefreet.so.1.7.99
b5f94000 b5f9d000 r-xp /usr/lib/libedbus.so.1.7.99
b5fa5000 b608a000 r-xp /usr/lib/libicuuc.so.51.1
b609f000 b61de000 r-xp /usr/lib/libicui18n.so.51.1
b61ee000 b624a000 r-xp /usr/lib/libedje.so.1.7.99
b6254000 b6265000 r-xp /usr/lib/libecore_input.so.1.7.99
b626d000 b6272000 r-xp /usr/lib/libecore_file.so.1.7.99
b627a000 b6293000 r-xp /usr/lib/libeet.so.1.7.99
b62a4000 b62a8000 r-xp /usr/lib/libappcore-common.so.1.1
b62b1000 b637d000 r-xp /usr/lib/libevas.so.1.7.99
b63a2000 b63c3000 r-xp /usr/lib/libecore_evas.so.1.7.99
b63cc000 b63fb000 r-xp /usr/lib/libecore_x.so.1.7.99
b6405000 b6539000 r-xp /usr/lib/libelementary.so.1.7.99
b6551000 b6552000 r-xp /usr/lib/libjournal.so.0.1.0
b655b000 b6626000 r-xp /usr/lib/libxml2.so.2.7.8
b6634000 b6644000 r-xp /lib/libresolv-2.13.so
b6648000 b665e000 r-xp /lib/libz.so.1.2.5
b6666000 b6668000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b6670000 b6675000 r-xp /usr/lib/libffi.so.5.0.10
b667e000 b667f000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6687000 b668a000 r-xp /lib/libattr.so.1.1.0
b6692000 b683a000 r-xp /usr/lib/libcrypto.so.1.0.0
b685a000 b6874000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b687d000 b68e6000 r-xp /lib/libm-2.13.so
b68ef000 b692f000 r-xp /usr/lib/libeina.so.1.7.99
b6938000 b6940000 r-xp /usr/lib/libvconf.so.0.2.45
b6948000 b694b000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6953000 b6987000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6990000 b6a64000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6a70000 b6a76000 r-xp /lib/librt-2.13.so
b6a7f000 b6a84000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6a8d000 b6a94000 r-xp /lib/libcrypt-2.13.so
b6ac4000 b6ac7000 r-xp /lib/libcap.so.2.21
b6acf000 b6ad1000 r-xp /usr/lib/libiri.so
b6ad9000 b6af8000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b00000 b6b16000 r-xp /usr/lib/libecore.so.1.7.99
b6b2c000 b6b31000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b3a000 b6c0a000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c0b000 b6c19000 r-xp /usr/lib/libail.so.0.1.0
b6c21000 b6c38000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c41000 b6c4b000 r-xp /lib/libunwind.so.8.0.1
b6c79000 b6d94000 r-xp /lib/libc-2.13.so
b6da2000 b6daa000 r-xp /lib/libgcc_s-4.6.4.so.1
b6db2000 b6ddc000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6de5000 b6de8000 r-xp /usr/lib/libbundle.so.0.1.22
b6df0000 b6df2000 r-xp /lib/libdl-2.13.so
b6dfb000 b6dfe000 r-xp /usr/lib/libsmack.so.1.0.0
b6e06000 b6e68000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6e72000 b6e84000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6e8d000 b6ea1000 r-xp /lib/libpthread-2.13.so
b6eae000 b6eb2000 r-xp /usr/lib/libappcore-efl.so.1.1
b6ebc000 b6ebe000 r-xp /usr/lib/libdlog.so.0.0.0
b6ec6000 b6ed1000 r-xp /usr/lib/libaul.so.0.1.0
b6edb000 b6edf000 r-xp /usr/lib/libsys-assert.so
b6ee8000 b6f05000 r-xp /lib/ld-2.13.so
b6f0e000 b6f14000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f1c000 b6f48000 rw-p [heap]
b6f48000 b71ae000 rw-p [heap]
beeda000 beefb000 rwxp [stack]
End of Maps Information

Callstack Information (PID:10539)
Call Stack Count: 9
 0: cfree + 0x30 (0xb6ce8150) [/lib/libc.so.6] + 0x6f150
 1: app_terminate + 0x2a (0xb3dbb76f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x976f
 2: (0xb44ca1e9) [/usr/lib/libcapi-appfw-application.so.0] + 0x11e9
 3: appcore_efl_main + 0x2c6 (0xb6eb0717) [/usr/lib/libappcore-efl.so.1] + 0x2717
 4: ui_app_main + 0xb0 (0xb44cac79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
 5: main + 0x10a (0xb3dbb90f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x990f
 6: (0xb6f10c6f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c6f
 7: __libc_start_main + 0x114 (0xb6c9082c) [/lib/libc.so.6] + 0x1782c
 8: (0xb6f1135c) [/opt/usr/apps/org.tizen.other/bin/other] + 0x335c
End of Call Stack

Package Information
Package Name: org.tizen.other
Package ID : org.tizen.other
Version: 1.0.0
Package Type: coretpk
App Name: Nornenjs
App ID: org.tizen.other
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
_LOG] __add_item_running_list pid: 2252
06-01 14:28:12.270+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.menu-screen
06-01 14:28:12.275+0900 D/AUL_AMD ( 2180): amd_launch.c: __reply_handler(784) > listen fd(28) , send fd(27), pid(2252), cmd(3)
06-01 14:28:12.275+0900 D/AUL     ( 2235): launch.c: app_request_to_launchpad(295) > launch request result : 2252
06-01 14:28:12.280+0900 D/STARTER ( 2235): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
06-01 14:28:12.280+0900 D/STARTER ( 2235): dbus-util.c: _dbus_message_send(172) > [_dbus_message_send:172] dbus_connection_send, ret=1
06-01 14:28:12.280+0900 E/STARTER ( 2235): dbus-util.c: starter_dbus_home_raise_signal_send(184) > [starter_dbus_home_raise_signal_send:184] Sending HOME RAISE signal, result:0
06-01 14:28:12.280+0900 D/RESOURCED( 2375): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2252, type = 0
06-01 14:28:12.280+0900 D/AUL_AMD ( 2180): amd_launch.c: __e17_status_handler(1888) > pid(2252) status(3)
06-01 14:28:12.280+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 2252
06-01 14:28:12.280+0900 D/RESOURCED( 2375): cpu.c: cpu_foreground_state(92) > [cpu_foreground_state,92] cpu_foreground_state : pid = 2252, appname = (null)
06-01 14:28:12.280+0900 D/RESOURCED( 2375): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/cgroup.procs, value 2252
06-01 14:28:12.360+0900 D/STARTER ( 2235): hw_key.c: _key_press_cb(656) > [_key_press_cb:656] _key_press_cb : XF86Phone Pressed
06-01 14:28:12.360+0900 W/STARTER ( 2235): hw_key.c: _key_press_cb(668) > [_key_press_cb:668] Home Key is pressed
06-01 14:28:12.360+0900 W/STARTER ( 2235): hw_key.c: _key_press_cb(686) > [_key_press_cb:686] homekey count : 1
06-01 14:28:12.360+0900 D/STARTER ( 2235): hw_key.c: _key_press_cb(695) > [_key_press_cb:695] create long press timer
06-01 14:28:12.480+0900 W/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2252
06-01 14:28:12.485+0900 D/indicator( 2229): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2600003"
06-01 14:28:12.485+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1a00003"
06-01 14:28:12.485+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
06-01 14:28:12.485+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1a00003"
06-01 14:28:12.485+0900 D/indicator( 2229): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
06-01 14:28:12.550+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(470) > [APP 2252] Event: MEM_FLUSH State: PAUSED
06-01 14:28:12.575+0900 D/APP_CORE(10539): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 1
06-01 14:28:12.575+0900 D/APP_CORE(10539): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
06-01 14:28:12.575+0900 D/APP_CORE(10539): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
06-01 14:28:12.575+0900 D/APP_CORE(10539): appcore-efl.c: __do_app(470) > [APP 10539] Event: PAUSE State: RUNNING
06-01 14:28:12.575+0900 D/RESOURCED( 2375): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 10539, type = 2
06-01 14:28:12.575+0900 D/APP_CORE(10539): appcore-efl.c: __do_app(538) > [APP 10539] PAUSE
06-01 14:28:12.575+0900 I/CAPI_APPFW_APPLICATION(10539): app_main.c: _ui_app_appcore_pause(603) > app_appcore_pause
06-01 14:28:12.575+0900 D/RESOURCED( 2375): cpu.c: cpu_background_state(100) > [cpu_background_state,100] cpu_background_state : pid = 10539, appname = other
06-01 14:28:12.575+0900 D/RESOURCED( 2375): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/service/cgroup.procs, value 10539
06-01 14:28:12.575+0900 D/APP_CORE(10539): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
06-01 14:28:12.580+0900 E/APP_CORE(10539): appcore-efl.c: __trm_app_info_send_socket(233) > access
06-01 14:28:12.580+0900 D/RESOURCED( 2375): proc-process.c: proc_backgrd_manage(145) > [proc_backgrd_manage,145] BACKGRD : process 2240 set score 420 (before 390)
06-01 14:28:12.580+0900 D/RESOURCED( 2375): proc-process.c: proc_backgrd_manage(145) > [proc_backgrd_manage,145] BACKGRD : process 9433 set score 330 (before 300)
06-01 14:28:12.580+0900 D/RESOURCED( 2375): proc-process.c: proc_backgrd_manage(152) > [proc_backgrd_manage,152] found candidate pid = -1091090952, count = 1
06-01 14:28:12.585+0900 I/RESOURCED( 2375): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/background/cgroup.procs, pid : 10539, oom : 300
06-01 14:28:12.585+0900 E/RESOURCED( 2375): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/background/cgroup.procs open failed
06-01 14:28:12.660+0900 D/STARTER ( 2235): hw_key.c: _destroy_syspopup_cb(630) > [_destroy_syspopup_cb:630] timer for cancel key operation
06-01 14:28:12.710+0900 D/APP_CORE( 2252): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1a00003 fully_obscured 0
06-01 14:28:12.710+0900 D/APP_CORE( 2252): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active 0
06-01 14:28:12.710+0900 D/APP_CORE( 2252): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
06-01 14:28:12.710+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(470) > [APP 2252] Event: RESUME State: PAUSED
06-01 14:28:12.710+0900 D/LAUNCH  ( 2252): appcore-efl.c: __do_app(557) > [menu-screen:Application:resume:start]
06-01 14:28:12.710+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(559) > [APP 2252] RESUME
06-01 14:28:12.710+0900 I/CAPI_APPFW_APPLICATION( 2252): app_main.c: app_appcore_resume(216) > app_appcore_resume
06-01 14:28:12.710+0900 D/MENU_SCREEN( 2252): menu_screen.c: _resume_cb(547) > START RESUME
06-01 14:28:12.710+0900 D/MENU_SCREEN( 2252): page_scroller.c: page_scroller_focus(1133) > Focus set scroller(0xb713d5e0), page:0, item:native
06-01 14:28:12.710+0900 D/LAUNCH  ( 2252): appcore-efl.c: __do_app(567) > [menu-screen:Application:resume:done]
06-01 14:28:12.710+0900 D/LAUNCH  ( 2252): appcore-efl.c: __do_app(569) > [menu-screen:Application:Launching:done]
06-01 14:28:12.710+0900 D/APP_CORE( 2252): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
06-01 14:28:12.710+0900 E/APP_CORE( 2252): appcore-efl.c: __trm_app_info_send_socket(233) > access
06-01 14:28:12.740+0900 I/SYSPOPUP( 2250): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
06-01 14:28:12.740+0900 I/SYSPOPUP( 2250): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
06-01 14:28:12.740+0900 D/VOLUME  ( 2250): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
06-01 14:28:12.740+0900 E/VOLUME  ( 2250): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
06-01 14:28:12.740+0900 E/VOLUME  ( 2250): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
06-01 14:28:12.740+0900 D/VOLUME  ( 2250): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
06-01 14:28:12.740+0900 D/VOLUME  ( 2250): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
06-01 14:28:12.740+0900 D/VOLUME  ( 2250): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
06-01 14:28:12.740+0900 E/VOLUME  ( 2250): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
06-01 14:28:12.740+0900 D/VOLUME  ( 2250): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
06-01 14:28:12.770+0900 D/STARTER ( 2235): hw_key.c: _launch_taskmgr_cb(132) > [_launch_taskmgr_cb:132] Launch TASKMGR
06-01 14:28:12.770+0900 D/STARTER ( 2235): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
06-01 14:28:12.770+0900 D/STARTER ( 2235): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
06-01 14:28:12.775+0900 D/AUL     ( 2235): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.task-mgr
06-01 14:28:12.775+0900 D/AUL     ( 2235): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(0)
06-01 14:28:12.775+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 0
06-01 14:28:12.780+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.task-mgr
06-01 14:28:12.785+0900 D/AUL     ( 2180): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 2232, pid = 2235
06-01 14:28:12.785+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
06-01 14:28:12.790+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1770) > process_pool: false
06-01 14:28:12.790+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1773) > h/w acceleration: USE
06-01 14:28:12.790+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.task-mgr
06-01 14:28:12.790+0900 D/AUL_AMD ( 2180): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
06-01 14:28:12.790+0900 D/AUL     ( 2180): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
06-01 14:28:12.790+0900 D/AUL_PAD ( 2217): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.task-mgr
06-01 14:28:12.790+0900 D/AUL_PAD ( 2217): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
06-01 14:28:12.790+0900 D/AUL_PAD ( 2217): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 10552 /usr/apps/org.tizen.task-mgr/bin/task-mgr
06-01 14:28:12.790+0900 D/AUL_PAD ( 2217): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
06-01 14:28:12.790+0900 D/AUL_PAD (10552): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
06-01 14:28:12.790+0900 D/AUL_PAD (10552): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
06-01 14:28:12.790+0900 D/AUL_PAD (10552): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.task-mgr / pkg_type : rpm / app_path : /usr/apps/org.tizen.task-mgr/bin/task-mgr 
06-01 14:28:12.805+0900 D/AUL_PAD (10552): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
06-01 14:28:12.805+0900 D/AUL_PAD (10552): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /usr/apps/org.tizen.task-mgr/bin/task-mgr##
06-01 14:28:12.805+0900 D/AUL_PAD (10552): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : HIDE_LAUNCH##
06-01 14:28:12.805+0900 D/AUL_PAD (10552): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_STARTTIME__##
06-01 14:28:12.805+0900 D/AUL_PAD (10552): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_PID__##
06-01 14:28:12.805+0900 D/LAUNCH  (10552): launchpad.c: __real_launch(229) > [SECURE_LOG] [/usr/apps/org.tizen.task-mgr/bin/task-mgr:Platform:launchpad:done]
06-01 14:28:12.805+0900 E/AUL_PAD (10552): preload.h: __preload_exec(124) > dlopen("/usr/apps/org.tizen.task-mgr/bin/task-mgr") failed
06-01 14:28:12.810+0900 E/AUL_PAD (10552): preload.h: __preload_exec(126) > dlopen error: /usr/apps/org.tizen.task-mgr/bin/task-mgr: cannot dynamically load executable
06-01 14:28:12.810+0900 D/AUL_PAD (10552): launchpad.c: __normal_fork_exec(190) > start real fork and exec
06-01 14:28:12.885+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: main(461) > Application Main Function 
06-01 14:28:12.885+0900 D/LAUNCH  (10552): appcore-efl.c: appcore_efl_main(1569) > [task-mgr:Application:main:done]
06-01 14:28:12.890+0900 D/AUL_PAD ( 2217): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
06-01 14:28:12.890+0900 D/AUL_PAD ( 2217): sigchild.h: __send_app_launch_signal(112) > send launch signal done
06-01 14:28:12.890+0900 D/AUL_PAD ( 2217): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
06-01 14:28:12.890+0900 D/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
06-01 14:28:12.890+0900 E/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(264) > access
06-01 14:28:12.890+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 10552
06-01 14:28:12.890+0900 D/RESOURCED( 2375): proc-noti.c: recv_str(87) > [recv_str,87] str is null
06-01 14:28:12.890+0900 D/RESOURCED( 2375): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2180
06-01 14:28:12.890+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.task-mgr, pid 10552, type 4 
06-01 14:28:12.890+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.task-mgr
06-01 14:28:12.890+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.task-mgr, 10552
06-01 14:28:12.890+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.task-mgr with pkgname
06-01 14:28:12.890+0900 D/RESOURCED( 2375): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 10552
06-01 14:28:12.890+0900 D/RESOURCED( 2375): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 10552
06-01 14:28:12.890+0900 D/RESOURCED( 2375): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
06-01 14:28:12.890+0900 D/RESOURCED( 2375): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
06-01 14:28:12.890+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
06-01 14:28:12.890+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
06-01 14:28:12.890+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
06-01 14:28:12.890+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
06-01 14:28:12.890+0900 E/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 540
06-01 14:28:12.890+0900 D/RESOURCED( 2375): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
06-01 14:28:12.890+0900 D/AUL     ( 2235): launch.c: app_request_to_launchpad(295) > launch request result : 10552
06-01 14:28:12.895+0900 D/STARTER ( 2235): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
06-01 14:28:12.900+0900 D/AUL     (10552): pkginfo.c: aul_app_get_appid_bypid(196) > [SECURE_LOG] appid for 10552 is org.tizen.task-mgr
06-01 14:28:13.030+0900 D/APP_CORE(10552): appcore-efl.c: __before_loop(1012) > elm_config_preferred_engine_set : opengl_x11
06-01 14:28:13.040+0900 D/AUL     (10552): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 10552 is org.tizen.task-mgr
06-01 14:28:13.040+0900 D/APP_CORE(10552): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.task-mgr/res/locale
06-01 14:28:13.040+0900 D/APP_CORE(10552): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
06-01 14:28:13.040+0900 D/AUL     (10552): app_sock.c: __create_server_sock(135) > pg path - already exists
06-01 14:28:13.040+0900 D/LAUNCH  (10552): appcore-efl.c: __before_loop(1035) > [task-mgr:Platform:appcore_init:done]
06-01 14:28:13.040+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: _create_app(331) > Application Create Callback 
06-01 14:28:13.190+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: create_layout(197) > create_layout
06-01 14:28:13.190+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: create_layout(216) > Resolution: HD
06-01 14:28:13.195+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: load_data(282) > load_data
06-01 14:28:13.195+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(625) > recent_app_list_create
06-01 14:28:13.195+0900 D/PKGMGR_INFO(10552): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3109) > [SECURE_LOG] where = package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
06-01 14:28:13.195+0900 D/PKGMGR_INFO(10552): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3115) > [SECURE_LOG] query = select DISTINCT package_app_info.*, package_app_localized_info.app_locale, package_app_localized_info.app_label, package_app_localized_info.app_icon from package_app_info LEFT OUTER JOIN package_app_localized_info ON package_app_info.app_id=package_app_localized_info.app_id and package_app_localized_info.app_locale IN ('No Locale', 'en-us') LEFT OUTER JOIN package_app_app_svc ON package_app_info.app_id=package_app_app_svc.app_id LEFT OUTER JOIN package_app_app_category ON package_app_info.app_id=package_app_app_category.app_id where package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
06-01 14:28:13.200+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 14:28:13.205+0900 D/AUL     (10552): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 409
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(653) > USP mode is disabled.
06-01 14:28:13.205+0900 E/RUA     (10552): rua.c: rua_history_load_db(278) > rua_history_load_db ok. nrows : 4, ncols : 5
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(680) > Apps in history: 4
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_retrieve_item(444) > [org.tizen.other]
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): [/opt/usr/apps/org.tizen.other/shared/res/favicon.png]
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): [Nornenjs]
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(741) > App added into the running_list : pkgid:org.tizen.other - appid:org.tizen.other
06-01 14:28:13.205+0900 E/TASK_MGR_LITE(10552): recent_apps.c: list_retrieve_item(348) > Fail to get ai table !!
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_retrieve_item(444) > [org.tizen.native]
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): [/opt/usr/apps/org.tizen.native/shared/res/native.png]
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): [native]
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(741) > App added into the running_list : pkgid:org.tizen.native - appid:org.tizen.native
06-01 14:28:13.205+0900 E/TASK_MGR_LITE(10552): recent_apps.c: list_retrieve_item(348) > Fail to get ai table !!
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(773) > HISTORY LIST: (nil) 0
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_create(774) > RUNNING LIST: 0x173ab0 2
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: load_data(292) > LISTs : 0x1be410, 0x1bf790
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: load_data(304) > App list should display !!
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_init(258) > in gen list: 0x173ab0 (nil)
06-01 14:28:13.205+0900 D/TASK_MGR_LITE(10552): genlist.c: recent_app_panel_create(98) > Creating task_mgr_genlist widget, noti count is : [-1]
06-01 14:28:13.220+0900 D/TASK_MGR_LITE(10552): genlist_item.c: genlist_item_class_create(308) > Genlist item class create
06-01 14:28:13.220+0900 D/TASK_MGR_LITE(10552): genlist_item.c: genlist_clear_all_class_create(323) > Genlist clear all class create
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_update(432) > genlist_update
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_clear_list(297) > genlist_clear_list
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: add_clear_btn_to_genlist(417) > add_clear_btn_to_genlist
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_add_item(165) > genlist_add_item
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_add_item(171) > Adding item: 0x1bd7c8
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid(210) > recent_apps_find_by_appid
06-01 14:28:13.235+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 14:28:13.235+0900 D/AUL     (10552): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 409
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.lockscreen (2240)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.volume (2250)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.menu-screen (2252)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.native (9433)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.other (10539)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(201) > FOUND
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.task-mgr (10552)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_add_item(165) > genlist_add_item
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_add_item(171) > Adding item: 0x1bb170
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid(210) > recent_apps_find_by_appid
06-01 14:28:13.235+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 14:28:13.235+0900 D/AUL     (10552): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 409
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.native <> org.tizen.lockscreen (2240)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.native <> org.tizen.volume (2250)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.native <> org.tizen.menu-screen (2252)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.native <> org.tizen.native (9433)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(201) > FOUND
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.native <> org.tizen.other (10539)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.native <> org.tizen.task-mgr (10552)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_update(449) > Are there recent apps: (nil)
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_update(450) > Adding recent apps... 0
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_item_show(192) > genlist_item_show
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_item_show(199) > ELM COUNT = 3
06-01 14:28:13.235+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: load_data(323) > load_data END. 
06-01 14:28:13.235+0900 D/LAUNCH  (10552): appcore-efl.c: __before_loop(1045) > [task-mgr:Application:create:done]
06-01 14:28:13.240+0900 D/APP_CORE(10552): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
06-01 14:28:13.240+0900 D/APP_CORE(10552): appcore.c: __aul_handler(423) > [APP 10552]     AUL event: AUL_START
06-01 14:28:13.240+0900 D/APP_CORE(10552): appcore-efl.c: __do_app(470) > [APP 10552] Event: RESET State: CREATED
06-01 14:28:13.240+0900 D/APP_CORE(10552): appcore-efl.c: __do_app(496) > [APP 10552] RESET
06-01 14:28:13.240+0900 D/LAUNCH  (10552): appcore-efl.c: __do_app(498) > [task-mgr:Application:reset:start]
06-01 14:28:13.240+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: _reset_app(446) > _reset_app
06-01 14:28:13.240+0900 D/LAUNCH  (10552): appcore-efl.c: __do_app(501) > [task-mgr:Application:reset:done]
06-01 14:28:13.240+0900 E/PKGMGR_INFO(10552): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_appinfo(1308) > (appid == NULL) appid is NULL
06-01 14:28:13.240+0900 I/APP_CORE(10552): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
06-01 14:28:13.240+0900 I/APP_CORE(10552): appcore-efl.c: __do_app(509) > [APP 10552] Initial Launching, call the resume_cb
06-01 14:28:13.240+0900 D/APP_CORE(10552): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
06-01 14:28:13.240+0900 D/APP_CORE(10552): appcore.c: __prt_ltime(183) > [APP 10552] first idle after reset: 466 msec
06-01 14:28:13.245+0900 D/APP_CORE(10552): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:3800008
06-01 14:28:13.245+0900 D/APP_CORE(10552): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:3800008
06-01 14:28:13.260+0900 W/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=10552
06-01 14:28:13.270+0900 D/indicator( 2229): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1a00003"
06-01 14:28:13.270+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 3800008"
06-01 14:28:13.270+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
06-01 14:28:13.270+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 3800008"
06-01 14:28:13.270+0900 D/indicator( 2229): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
06-01 14:28:13.275+0900 D/AUL_AMD ( 2180): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.menu-screen /usr/apps/org.tizen.menu-screen/bin/menu-screen
06-01 14:28:13.275+0900 D/RUA     ( 2180): rua.c: rua_add_history(179) > rua_add_history start
06-01 14:28:13.290+0900 D/RUA     ( 2180): rua.c: rua_add_history(247) > rua_add_history ok
06-01 14:28:13.325+0900 D/AUL_AMD ( 2180): amd_launch.c: __e17_status_handler(1888) > pid(10552) status(3)
06-01 14:28:13.325+0900 D/RESOURCED( 2375): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 10552, type = 0
06-01 14:28:13.325+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 10552
06-01 14:28:13.325+0900 D/RESOURCED( 2375): cpu.c: cpu_foreground_state(92) > [cpu_foreground_state,92] cpu_foreground_state : pid = 10552, appname = (null)
06-01 14:28:13.325+0900 D/RESOURCED( 2375): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/cgroup.procs, value 10552
06-01 14:28:13.340+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
06-01 14:28:13.345+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
06-01 14:28:13.345+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/favicon.png; Nornenjs 0x21e2e8
06-01 14:28:13.345+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(169) > It is already set.
06-01 14:28:13.345+0900 D/TASK_MGR_LITE(10552): genlist_item.c: recent_app_item_create(543) > 
06-01 14:28:13.345+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_main_create(765) > 
06-01 14:28:13.350+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
06-01 14:28:13.350+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
06-01 14:28:13.355+0900 E/TASK_MGR_LITE(10552): genlist_item.c: del_cb(758) > Deleted
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.native/shared/res/native.png; native 0x21e438
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(169) > It is already set.
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: recent_app_item_create(543) > 
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_main_create(765) > 
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
06-01 14:28:13.355+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
06-01 14:28:13.355+0900 E/TASK_MGR_LITE(10552): genlist_item.c: del_cb(758) > Deleted
06-01 14:28:13.355+0900 D/APP_CORE(10552): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:3800008 fully_obscured 0
06-01 14:28:13.355+0900 D/APP_CORE(10552): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
06-01 14:28:13.355+0900 D/APP_CORE(10552): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
06-01 14:28:13.355+0900 D/APP_CORE(10552): appcore-efl.c: __do_app(470) > [APP 10552] Event: RESUME State: RUNNING
06-01 14:28:13.360+0900 D/LAUNCH  (10552): appcore-efl.c: __do_app(557) > [task-mgr:Application:resume:start]
06-01 14:28:13.360+0900 D/LAUNCH  (10552): appcore-efl.c: __do_app(567) > [task-mgr:Application:resume:done]
06-01 14:28:13.360+0900 D/LAUNCH  (10552): appcore-efl.c: __do_app(569) > [task-mgr:Application:Launching:done]
06-01 14:28:13.360+0900 D/APP_CORE(10552): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
06-01 14:28:13.360+0900 E/APP_CORE(10552): appcore-efl.c: __trm_app_info_send_socket(233) > access
06-01 14:28:13.360+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
06-01 14:28:13.360+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
06-01 14:28:13.360+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/favicon.png; Nornenjs 0x21e2e8
06-01 14:28:13.360+0900 D/TASK_MGR_LITE(10552): genlist_item.c: recent_app_item_create(543) > 
06-01 14:28:13.360+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_main_create(765) > 
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.native/shared/res/native.png; native 0x21e438
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: recent_app_item_create(543) > 
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_main_create(765) > 
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
06-01 14:28:13.365+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
06-01 14:28:13.615+0900 D/STARTER ( 2235): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
06-01 14:28:13.615+0900 W/STARTER ( 2235): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
06-01 14:28:13.615+0900 D/STARTER ( 2235): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
06-01 14:28:13.615+0900 D/STARTER ( 2235): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
06-01 14:28:13.625+0900 D/STARTER ( 2235): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
06-01 14:28:13.690+0900 I/SYSPOPUP( 2250): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
06-01 14:28:13.695+0900 I/SYSPOPUP( 2250): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
06-01 14:28:13.695+0900 D/VOLUME  ( 2250): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
06-01 14:28:13.695+0900 E/VOLUME  ( 2250): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
06-01 14:28:13.695+0900 E/VOLUME  ( 2250): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
06-01 14:28:13.695+0900 D/VOLUME  ( 2250): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
06-01 14:28:13.695+0900 D/VOLUME  ( 2250): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
06-01 14:28:13.695+0900 D/VOLUME  ( 2250): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
06-01 14:28:13.695+0900 E/VOLUME  ( 2250): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
06-01 14:28:13.695+0900 D/VOLUME  ( 2250): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
06-01 14:28:13.900+0900 D/AUL_AMD ( 2180): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.task-mgr /usr/apps/org.tizen.task-mgr/bin/task-mgr
06-01 14:28:13.900+0900 D/RUA     ( 2180): rua.c: rua_add_history(179) > rua_add_history start
06-01 14:28:13.910+0900 D/RUA     ( 2180): rua.c: rua_add_history(247) > rua_add_history ok
06-01 14:28:14.070+0900 D/EFL     (10552): ecore_x<10552> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=42564196 button=1
06-01 14:28:14.085+0900 D/EFL     (10552): ecore_x<10552> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=42564218 button=1
06-01 14:28:14.085+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x601bcd 
06-01 14:28:14.085+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
06-01 14:28:14.085+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected list 0.000000 no_shutdown: 0
06-01 14:28:14.085+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
06-01 14:28:14.085+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected layout 0.000000 no_shutdown: 0
06-01 14:28:14.085+0900 D/TASK_MGR_LITE(10552): genlist_item.c: clear_all_btn_clicked_cb(218) > Removing all items...
06-01 14:28:14.090+0900 D/TASK_MGR_LITE(10552): genlist_item.c: clear_all_btn_clicked_cb(230) > On REDWOOD target (HD) : No Animation.
06-01 14:28:14.090+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_apps_kill_all(164) > recent_apps_kill_all
06-01 14:28:14.090+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 14:28:14.090+0900 D/AUL     (10552): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 409
06-01 14:28:14.095+0900 D/TASK_MGR_LITE(10552): recent_apps.c: get_app_taskmanage(121) > app org.tizen.lockscreen is taskmanage: 0
06-01 14:28:14.105+0900 D/TASK_MGR_LITE(10552): recent_apps.c: get_app_taskmanage(121) > app org.tizen.volume is taskmanage: 0
06-01 14:28:14.110+0900 D/TASK_MGR_LITE(10552): recent_apps.c: get_app_taskmanage(121) > app org.tizen.menu-screen is taskmanage: 0
06-01 14:28:14.115+0900 D/TASK_MGR_LITE(10552): recent_apps.c: get_app_taskmanage(121) > app org.tizen.native is taskmanage: 1
06-01 14:28:14.115+0900 D/AUL     (10552): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : 9433
06-01 14:28:14.115+0900 D/AUL     (10552): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(4)
06-01 14:28:14.115+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 4
06-01 14:28:14.125+0900 D/RESOURCED( 2375): proc-noti.c: recv_str(87) > [recv_str,87] str is null
06-01 14:28:14.125+0900 D/RESOURCED( 2375): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2180
06-01 14:28:14.125+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid (null), pid 9433, type 6 
06-01 14:28:14.125+0900 E/RESOURCED( 2375): datausage-common.c: app_terminate_cb(254) > [app_terminate_cb,254] No classid to terminate!
06-01 14:28:14.125+0900 D/RESOURCED( 2375): proc-main.c: proc_remove_process_list(363) > [proc_remove_process_list,363] found_pid 9433
06-01 14:28:14.130+0900 D/AUL_AMD ( 2180): amd_request.c: __app_process_by_pid(175) > __app_process_by_pid, cmd: 4, pid: 9433, 
06-01 14:28:14.130+0900 D/AUL     ( 2180): app_sock.c: __app_send_raw_with_delay_reply(434) > pid(9433) : cmd(4)
06-01 14:28:14.135+0900 D/AUL_AMD ( 2180): amd_launch.c: _term_app(878) > term done
06-01 14:28:14.135+0900 D/AUL_AMD ( 2180): amd_launch.c: __set_reply_handler(860) > listen fd : 28, send fd : 27
06-01 14:28:14.135+0900 D/APP_CORE( 9433): appcore.c: __aul_handler(443) > [APP 9433]     AUL event: AUL_TERMINATE
06-01 14:28:14.135+0900 D/APP_CORE( 9433): appcore-efl.c: __do_app(470) > [APP 9433] Event: TERMINATE State: PAUSED
06-01 14:28:14.135+0900 D/APP_CORE( 9433): appcore-efl.c: __do_app(486) > [APP 9433] TERMINATE
06-01 14:28:14.135+0900 D/AUL     ( 9433): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
06-01 14:28:14.135+0900 I/CAPI_APPFW_APPLICATION( 9433): app_main.c: _ui_app_appcore_terminate(577) > app_appcore_terminate
06-01 14:28:14.145+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 22
06-01 14:28:14.145+0900 D/AUL_AMD ( 2180): amd_launch.c: __reply_handler(784) > listen fd(28) , send fd(27), pid(9433), cmd(4)
06-01 14:28:14.145+0900 D/AUL     (10552): launch.c: app_request_to_launchpad(295) > launch request result : 0
06-01 14:28:14.145+0900 D/TASK_MGR_LITE(10552): recent_apps.c: kill_pid(141) > terminate pid = 9433
06-01 14:28:14.150+0900 D/TASK_MGR_LITE(10552): recent_apps.c: get_app_taskmanage(121) > app org.tizen.other is taskmanage: 1
06-01 14:28:14.150+0900 D/AUL     (10552): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : 10539
06-01 14:28:14.150+0900 D/AUL     (10552): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(4)
06-01 14:28:14.150+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 4
06-01 14:28:14.150+0900 D/RESOURCED( 2375): proc-noti.c: recv_str(87) > [recv_str,87] str is null
06-01 14:28:14.155+0900 D/RESOURCED( 2375): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2180
06-01 14:28:14.155+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid (null), pid 10539, type 6 
06-01 14:28:14.155+0900 E/RESOURCED( 2375): datausage-common.c: app_terminate_cb(254) > [app_terminate_cb,254] No classid to terminate!
06-01 14:28:14.155+0900 D/RESOURCED( 2375): proc-main.c: proc_remove_process_list(363) > [proc_remove_process_list,363] found_pid 10539
06-01 14:28:14.155+0900 D/AUL_AMD ( 2180): amd_request.c: __app_process_by_pid(175) > __app_process_by_pid, cmd: 4, pid: 10539, 
06-01 14:28:14.155+0900 D/AUL     ( 2180): app_sock.c: __app_send_raw_with_delay_reply(434) > pid(10539) : cmd(4)
06-01 14:28:14.155+0900 D/AUL_AMD ( 2180): amd_launch.c: _term_app(878) > term done
06-01 14:28:14.155+0900 D/AUL_AMD ( 2180): amd_launch.c: __set_reply_handler(860) > listen fd : 28, send fd : 27
06-01 14:28:14.185+0900 D/AUL_AMD ( 2180): amd_launch.c: __reply_handler(784) > listen fd(28) , send fd(27), pid(10539), cmd(4)
06-01 14:28:14.185+0900 D/AUL     (10552): launch.c: app_request_to_launchpad(295) > launch request result : 0
06-01 14:28:14.185+0900 D/TASK_MGR_LITE(10552): recent_apps.c: kill_pid(141) > terminate pid = 10539
06-01 14:28:14.185+0900 D/APP_CORE(10539): appcore.c: __aul_handler(443) > [APP 10539]     AUL event: AUL_TERMINATE
06-01 14:28:14.185+0900 D/APP_CORE(10539): appcore-efl.c: __do_app(470) > [APP 10539] Event: TERMINATE State: PAUSED
06-01 14:28:14.185+0900 D/APP_CORE(10539): appcore-efl.c: __do_app(486) > [APP 10539] TERMINATE
06-01 14:28:14.185+0900 D/AUL     (10539): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
06-01 14:28:14.185+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 22
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): recent_apps.c: get_app_taskmanage(121) > app org.tizen.task-mgr is taskmanage: 0
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_del(189) > _recent_app_item_content_del
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_del(194) > Data exist: 0xb0b13b50
06-01 14:28:14.190+0900 E/TASK_MGR_LITE(10552): genlist_item.c: del_cb(758) > Deleted
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_del(198) > Item deleted
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_del(189) > _recent_app_item_content_del
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_del(194) > Data exist: 0xb0b08648
06-01 14:28:14.190+0900 E/TASK_MGR_LITE(10552): genlist_item.c: del_cb(758) > Deleted
06-01 14:28:14.190+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _recent_app_item_content_del(198) > Item deleted
06-01 14:28:14.195+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _clear_all_button_del(206) > _clear_all_button_del
06-01 14:28:14.195+0900 D/TASK_MGR_LITE(10552): genlist_item.c: _clear_all_button_del(213) > _clear_all_button_del END
06-01 14:28:14.220+0900 D/AUL     (10552): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
06-01 14:28:14.220+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 22
06-01 14:28:14.220+0900 D/APP_CORE(10552): appcore-efl.c: __after_loop(1060) > [APP 10552] PAUSE before termination
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: _pause_app(453) > _pause_app
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: _terminate_app(393) > _terminate_app
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): genlist.c: genlist_clear_list(297) > genlist_clear_list
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): genlist.c: recent_app_panel_del_cb(84) > recent_app_panel_del_cb
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): genlist_item.c: genlist_item_class_destroy(340) > Genlist class free
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): task-mgr-lite.c: delete_layout(272) > delete_layout
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: recent_app_list_destroy(791) > recent_app_list_destroy
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_destroy(475) > START list_destroy
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_destroy(487) > FREE ALL list_destroy
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_destroy(493) > FREING: list_destroy org.tizen.other
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_unretrieve_item(295) > FREING 0x1bd7c8 org.tizen.other 's Item(list_type_default_s) in list_unretrieve_item
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_unretrieve_item(333) > list_unretrieve_item END 
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_destroy(493) > FREING: list_destroy org.tizen.native
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_unretrieve_item(295) > FREING 0x1bb170 org.tizen.native 's Item(list_type_default_s) in list_unretrieve_item
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_unretrieve_item(333) > list_unretrieve_item END 
06-01 14:28:14.225+0900 D/TASK_MGR_LITE(10552): recent_apps.c: list_destroy(500) > END list_destroy
06-01 14:28:14.235+0900 E/APP_CORE(10552): appcore.c: appcore_flush_memory(583) > Appcore not initialized
06-01 14:28:14.235+0900 W/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2252
06-01 14:28:14.245+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_wininfo_del(141) > [PROCESSMGR] delete anr_trigger_timer!
06-01 14:28:14.245+0900 D/indicator( 2229): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 3800008"
06-01 14:28:14.245+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1a00003"
06-01 14:28:14.255+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
06-01 14:28:14.260+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1a00003"
06-01 14:28:14.260+0900 D/indicator( 2229): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
06-01 14:28:14.380+0900 I/CAPI_APPFW_APPLICATION(10539): app_main.c: _ui_app_appcore_terminate(577) > app_appcore_terminate
06-01 14:28:14.580+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=42564708 button=1
06-01 14:28:14.580+0900 D/MENU_SCREEN( 2252): mouse.c: _down_cb(103) > Mouse down (354,855)
06-01 14:28:14.645+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x600056 
06-01 14:28:14.645+0900 D/EFL     ( 2252): ecore_x<2252> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=42564775 button=1
06-01 14:28:14.645+0900 D/MENU_SCREEN( 2252): mouse.c: _up_cb(120) > Mouse up (356,849)
06-01 14:28:14.810+0900 W/CRASH_MANAGER(10558): worker.c: worker_job(1189) > 11105396f7468143313649
