S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 23736
Date: 2015-06-01 20:54:58+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xaf1a3004

Register Information
r0   = 0xaf1a3008, r1   = 0xb3dbea17
r2   = 0x000000e4, r3   = 0x00000000
r4   = 0xb3e658ec, r5   = 0xaf1a3008
r6   = 0x00000241, r7   = 0xbeefa000
r8   = 0xb6eba2e8, r9   = 0xb6eba2e4
r10  = 0x00000000, fp   = 0xb6f11f40
ip   = 0xb3e65968, sp   = 0xbeef9fe8
lr   = 0xb3dbea17, pc   = 0xb6ce8150
cpsr = 0xa0000010

Memory Information
MemTotal:   797840 KB
MemFree:    410640 KB
Buffers:     42248 KB
Cached:     134260 KB
VmPeak:     149696 KB
VmSize:     119208 KB
VmLck:           0 KB
VmHWM:       16292 KB
VmRSS:       16292 KB
VmData:      63392 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24908 KB
VmPTE:          88 KB
VmSwap:          0 KB

Threads Information
Threads: 6
PID = 23736 TID = 23736
23736 23741 23742 23743 23745 23746 

Maps Information
b175d000 b175e000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17a6000 b17a7000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17af000 b17b6000 r-xp /usr/lib/libfeedback.so.0.1.4
b17c9000 b17ca000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b17d2000 b17e9000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b196f000 b1974000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31bd000 b3208000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b3211000 b321b000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3224000 b3280000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3a8d000 b3a93000 r-xp /usr/lib/libUMP.so
b3a9b000 b3aae000 r-xp /usr/lib/libEGL_platform.so
b3ab7000 b3b8e000 r-xp /usr/lib/libMali.so
b3b99000 b3bb0000 r-xp /usr/lib/libEGL.so.1.4
b3bb9000 b3bbe000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3bc7000 b3bc8000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3bd1000 b3be9000 r-xp /usr/lib/libpng12.so.0.50.0
b3bf1000 b3c2f000 r-xp /usr/lib/libGLESv2.so.2.0
b3c37000 b3c3b000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c44000 b3c46000 r-xp /usr/lib/libdri2.so.0.0.0
b3c4e000 b3c55000 r-xp /usr/lib/libdrm.so.2.4.0
b3c5e000 b3d14000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d1f000 b3d35000 r-xp /usr/lib/libtts.so
b3d3e000 b3d45000 r-xp /usr/lib/libtbm.so.1.0.0
b3d4d000 b3d52000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3d5a000 b3d6b000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3d73000 b3d7a000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3d82000 b3d87000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3d8f000 b3d91000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3d99000 b3da1000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3da9000 b3dac000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3db5000 b3e5c000 r-xp /opt/usr/apps/org.tizen.other/bin/other
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
b6f48000 b716e000 rw-p [heap]
beeda000 beefb000 rwxp [stack]
End of Maps Information

Callstack Information (PID:23736)
Call Stack Count: 9
 0: cfree + 0x30 (0xb6ce8150) [/lib/libc.so.6] + 0x6f150
 1: app_terminate + 0x2a (0xb3dbea17) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9a17
 2: (0xb44ca1e9) [/usr/lib/libcapi-appfw-application.so.0] + 0x11e9
 3: appcore_efl_main + 0x2c6 (0xb6eb0717) [/usr/lib/libappcore-efl.so.1] + 0x2717
 4: ui_app_main + 0xb0 (0xb44cac79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
 5: main + 0x10a (0xb3dbebb7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9bb7
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
1 20:48:48.240+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
06-01 20:48:48.240+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
06-01 20:48:48.240+0900 D/AUL_AMD ( 2180): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
06-01 20:48:48.240+0900 D/AUL     ( 2180): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
06-01 20:48:48.240+0900 D/AUL_PAD ( 2217): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
06-01 20:48:48.240+0900 D/AUL_PAD ( 2217): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
06-01 20:48:48.245+0900 D/AUL_PAD ( 2217): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 23736 /opt/usr/apps/org.tizen.other/bin/other
06-01 20:48:48.245+0900 D/AUL_PAD (23736): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
06-01 20:48:48.245+0900 D/AUL_PAD ( 2217): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
06-01 20:48:48.245+0900 D/AUL_PAD (23736): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
06-01 20:48:48.245+0900 D/AUL_PAD (23736): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
06-01 20:48:48.265+0900 D/AUL_PAD (23736): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
06-01 20:48:48.265+0900 D/AUL_PAD (23736): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
06-01 20:48:48.265+0900 D/AUL_PAD (23736): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
06-01 20:48:48.265+0900 D/AUL_PAD (23736): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
06-01 20:48:48.265+0900 D/LAUNCH  (23736): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
06-01 20:48:48.305+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: ui_app_main(697) > app_efl_main
06-01 20:48:48.305+0900 D/LAUNCH  (23736): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
06-01 20:48:48.345+0900 D/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
06-01 20:48:48.345+0900 E/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(264) > access
06-01 20:48:48.345+0900 D/RESOURCED( 2375): proc-noti.c: recv_str(87) > [recv_str,87] str is null
06-01 20:48:48.345+0900 D/RESOURCED( 2375): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2180
06-01 20:48:48.345+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 23736, type 4 
06-01 20:48:48.345+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 23736
06-01 20:48:48.345+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
06-01 20:48:48.345+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 23736
06-01 20:48:48.345+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
06-01 20:48:48.345+0900 D/RESOURCED( 2375): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 23736
06-01 20:48:48.345+0900 D/RESOURCED( 2375): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 23736
06-01 20:48:48.345+0900 D/RESOURCED( 2375): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
06-01 20:48:48.345+0900 D/RESOURCED( 2375): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
06-01 20:48:48.345+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
06-01 20:48:48.345+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
06-01 20:48:48.345+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
06-01 20:48:48.345+0900 D/RESOURCED( 2375): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
06-01 20:48:48.350+0900 E/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 552
06-01 20:48:48.350+0900 D/RESOURCED( 2375): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
06-01 20:48:48.350+0900 D/AUL_PAD ( 2217): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
06-01 20:48:48.350+0900 D/AUL_PAD ( 2217): sigchild.h: __send_app_launch_signal(112) > send launch signal done
06-01 20:48:48.350+0900 D/AUL_PAD ( 2217): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
06-01 20:48:48.365+0900 D/APP_CORE(23736): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
06-01 20:48:48.375+0900 D/AUL     (23736): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 23736 is org.tizen.other
06-01 20:48:48.375+0900 D/APP_CORE(23736): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
06-01 20:48:48.375+0900 D/APP_CORE(23736): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
06-01 20:48:48.375+0900 D/AUL     (23736): app_sock.c: __create_server_sock(135) > pg path - already exists
06-01 20:48:48.375+0900 D/LAUNCH  (23736): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
06-01 20:48:48.375+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
06-01 20:48:48.590+0900 W/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=23736
06-01 20:48:48.645+0900 D/indicator( 2229): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1a00003"
06-01 20:48:48.655+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
06-01 20:48:48.655+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
06-01 20:48:48.655+0900 D/indicator( 2229): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
06-01 20:48:48.655+0900 D/indicator( 2229): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
06-01 20:48:48.690+0900 F/socket.io(23736): thread_start
06-01 20:48:48.690+0900 F/socket.io(23736): finish 0
06-01 20:48:48.690+0900 D/LAUNCH  (23736): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
06-01 20:48:48.690+0900 D/APP_CORE(23736): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
06-01 20:48:48.690+0900 D/APP_CORE(23736): appcore.c: __aul_handler(423) > [APP 23736]     AUL event: AUL_START
06-01 20:48:48.690+0900 D/APP_CORE(23736): appcore-efl.c: __do_app(470) > [APP 23736] Event: RESET State: CREATED
06-01 20:48:48.690+0900 D/APP_CORE(23736): appcore-efl.c: __do_app(496) > [APP 23736] RESET
06-01 20:48:48.690+0900 D/LAUNCH  (23736): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
06-01 20:48:48.690+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
06-01 20:48:48.690+0900 D/APP_SVC (23736): appsvc.c: __set_bundle(161) > __set_bundle
06-01 20:48:48.690+0900 D/LAUNCH  (23736): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
06-01 20:48:48.700+0900 I/APP_CORE(23736): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
06-01 20:48:48.700+0900 I/APP_CORE(23736): appcore-efl.c: __do_app(509) > [APP 23736] Initial Launching, call the resume_cb
06-01 20:48:48.700+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
06-01 20:48:48.700+0900 D/APP_CORE(23736): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
06-01 20:48:48.700+0900 D/APP_CORE(23736): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600003
06-01 20:48:48.700+0900 D/APP_CORE(23736): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600003
06-01 20:48:48.720+0900 E/socket.io(23736): 566: Connected.
06-01 20:48:48.720+0900 E/socket.io(23736): 554: On handshake, sid
06-01 20:48:48.720+0900 E/socket.io(23736): 651: Received Message type(connect)
06-01 20:48:48.720+0900 E/socket.io(23736): 489: On Connected
06-01 20:48:48.720+0900 F/sio_packet(23736): accept()
06-01 20:48:48.720+0900 E/socket.io(23736): 743: encoded paylod length: 18
06-01 20:48:48.720+0900 E/socket.io(23736): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 20:48:48.810+0900 E/socket.io(23736): 669: Received Message type(Event)
06-01 20:48:48.810+0900 F/sio_packet(23736): accept()
06-01 20:48:48.810+0900 E/socket.io(23736): 743: encoded paylod length: 21
06-01 20:48:48.810+0900 E/socket.io(23736): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 20:48:48.840+0900 D/APP_CORE( 2252): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1a00003 fully_obscured 1
06-01 20:48:48.840+0900 D/APP_CORE( 2252): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
06-01 20:48:48.840+0900 D/APP_CORE( 2252): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
06-01 20:48:48.840+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(470) > [APP 2252] Event: PAUSE State: RUNNING
06-01 20:48:48.840+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(538) > [APP 2252] PAUSE
06-01 20:48:48.840+0900 I/CAPI_APPFW_APPLICATION( 2252): app_main.c: app_appcore_pause(195) > app_appcore_pause
06-01 20:48:48.840+0900 D/MENU_SCREEN( 2252): menu_screen.c: _pause_cb(538) > Pause start
06-01 20:48:48.840+0900 D/APP_CORE( 2252): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
06-01 20:48:48.840+0900 E/APP_CORE( 2252): appcore-efl.c: __trm_app_info_send_socket(233) > access
06-01 20:48:48.845+0900 D/APP_CORE(23736): appcore.c: __prt_ltime(183) > [APP 23736] first idle after reset: 626 msec
06-01 20:48:48.845+0900 D/APP_CORE(23736): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 0
06-01 20:48:48.845+0900 D/APP_CORE(23736): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
06-01 20:48:48.845+0900 D/APP_CORE(23736): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
06-01 20:48:48.845+0900 D/APP_CORE(23736): appcore-efl.c: __do_app(470) > [APP 23736] Event: RESUME State: RUNNING
06-01 20:48:48.845+0900 D/LAUNCH  (23736): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
06-01 20:48:48.845+0900 D/LAUNCH  (23736): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
06-01 20:48:48.845+0900 D/LAUNCH  (23736): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
06-01 20:48:48.845+0900 D/APP_CORE(23736): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
06-01 20:48:48.845+0900 E/APP_CORE(23736): appcore-efl.c: __trm_app_info_send_socket(233) > access
06-01 20:48:48.845+0900 D/RESOURCED( 2375): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2252, type = 2
06-01 20:48:48.850+0900 D/RESOURCED( 2375): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 23736, type = 0
06-01 20:48:48.850+0900 D/AUL_AMD ( 2180): amd_launch.c: __e17_status_handler(1888) > pid(23736) status(3)
06-01 20:48:48.850+0900 D/RESOURCED( 2375): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 23736
06-01 20:48:48.850+0900 I/RESOURCED( 2375): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 23736, oom : 200
06-01 20:48:48.850+0900 E/RESOURCED( 2375): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
06-01 20:48:48.860+0900 E/socket.io(23736): 669: Received Message type(Event)
06-01 20:48:48.860+0900 F/get_binary(23736): in get binary_message()...
06-01 20:48:48.880+0900 D/CAPI_MEDIA_IMAGE_UTIL(23736): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 20:48:48.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:48:48.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:48:49.360+0900 D/AUL_AMD ( 2180): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.other /opt/usr/apps/org.tizen.other/bin/other
06-01 20:48:49.360+0900 D/RUA     ( 2180): rua.c: rua_add_history(179) > rua_add_history start
06-01 20:48:49.375+0900 D/RUA     ( 2180): rua.c: rua_add_history(247) > rua_add_history ok
06-01 20:48:49.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:48:49.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:48:50.450+0900 D/EFL     (23736): ecore_x<23736> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=65241486 button=1
06-01 20:48:50.760+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x65a090 
06-01 20:48:50.770+0900 D/EFL     (23736): ecore_x<23736> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=65241794 button=1
06-01 20:48:50.840+0900 D/EFL     (23736): ecore_x<23736> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=65241859 button=1
06-01 20:48:50.870+0900 D/EFL     (23736): ecore_x<23736> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=65241902 button=1
06-01 20:48:50.880+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x65a090 
06-01 20:48:51.880+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_begin_handler(406) > [PROCESSMGR] ecore_x_netwm_ping_send to the client_win=0x2600003
06-01 20:48:53.910+0900 D/APP_CORE( 2252): appcore-efl.c: __do_app(470) > [APP 2252] Event: MEM_FLUSH State: PAUSED
06-01 20:48:53.925+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:48:56.880+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_confirm_handler(336) > [PROCESSMGR] last_pointed_win=0x65a090 bd->visible=1
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 20:49 5 HH:mm"
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 20:49"
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 20&#x2236;49"
06-01 20:49:00.960+0900 D/indicator( 2229): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 367184 Time: <font_size=34>20&#x2236;49</font_size></font>"
06-01 20:49:13.810+0900 F/sio_packet(23736): accept()
06-01 20:49:13.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:13.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:49:14.475+0900 D/EFL     (23736): ecore_x<23736> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=65265511 button=1
06-01 20:49:14.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:14.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:49:15.175+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x65a090 
06-01 20:49:15.185+0900 D/EFL     (23736): ecore_x<23736> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=65266210 button=1
06-01 20:49:16.175+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_begin_handler(406) > [PROCESSMGR] ecore_x_netwm_ping_send to the client_win=0x2600003
06-01 20:49:21.175+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_confirm_handler(336) > [PROCESSMGR] last_pointed_win=0x65a090 bd->visible=1
06-01 20:49:33.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:33.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:49:34.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:34.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:49:38.810+0900 F/sio_packet(23736): accept()
06-01 20:49:38.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:38.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:49:39.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:39.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:49:44.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:44.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:49:45.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:49:45.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:49:53.975+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:50:00.970+0900 D/indicator( 2229): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
06-01 20:50:00.975+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
06-01 20:50:00.975+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
06-01 20:50:00.975+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 20:50 5 HH:mm"
06-01 20:50:00.975+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 20:50"
06-01 20:50:00.975+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 20&#x2236;50"
06-01 20:50:00.975+0900 D/indicator( 2229): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 366928 Time: <font_size=34>20&#x2236;50</font_size></font>"
06-01 20:50:03.810+0900 F/sio_packet(23736): accept()
06-01 20:50:03.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:03.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:50:04.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:04.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:50:23.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:23.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:50:25.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:25.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:50:28.810+0900 F/sio_packet(23736): accept()
06-01 20:50:28.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:28.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:50:29.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:29.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:50:44.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:44.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:50:45.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:45.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:50:53.810+0900 F/sio_packet(23736): accept()
06-01 20:50:53.975+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:50:53.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:53.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:50:54.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:50:54.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:00.975+0900 D/indicator( 2229): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
06-01 20:51:00.975+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
06-01 20:51:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
06-01 20:51:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 20:51 5 HH:mm"
06-01 20:51:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 20:51"
06-01 20:51:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 20&#x2236;51"
06-01 20:51:00.980+0900 D/indicator( 2229): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 367056 Time: <font_size=34>20&#x2236;51</font_size></font>"
06-01 20:51:10.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:10.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:51:12.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:12.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:13.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:13.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:51:14.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:14.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:17.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:17.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:51:18.810+0900 F/sio_packet(23736): accept()
06-01 20:51:18.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:18.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:51:19.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:19.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:21.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:21.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:51:26.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:26.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:41.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:41.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:51:42.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:42.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:43.810+0900 F/sio_packet(23736): accept()
06-01 20:51:43.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:43.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:51:44.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:44.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:51:45.985+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:45.985+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:51:53.975+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:51:55.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:51:55.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 20:52 5 HH:mm"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 20:52"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 20&#x2236;52"
06-01 20:52:00.980+0900 D/indicator( 2229): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 366928 Time: <font_size=34>20&#x2236;52</font_size></font>"
06-01 20:52:08.810+0900 F/sio_packet(23736): accept()
06-01 20:52:08.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:08.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:52:09.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:09.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:52:18.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:18.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:52:22.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:22.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:52:24.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:24.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:52:25.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:25.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:52:33.810+0900 F/sio_packet(23736): accept()
06-01 20:52:33.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:33.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:52:34.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:34.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:52:53.975+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:52:58.815+0900 F/sio_packet(23736): accept()
06-01 20:52:58.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:58.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:52:59.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:52:59.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 20:53 5 HH:mm"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 20:53"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 20&#x2236;53"
06-01 20:53:00.980+0900 D/indicator( 2229): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 367184 Time: <font_size=34>20&#x2236;53</font_size></font>"
06-01 20:53:04.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:04.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:07.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:07.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:23.815+0900 F/sio_packet(23736): accept()
06-01 20:53:23.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:23.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:53:24.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:24.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:25.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:25.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:26.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:26.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:30.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:30.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:34.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:34.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:36.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:36.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:37.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:37.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:38.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:38.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:40.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:40.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:44.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:44.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:53:45.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:45.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:48.815+0900 F/sio_packet(23736): accept()
06-01 20:53:48.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:48.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:53:49.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:49.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:53:53.975+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:53:55.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:53:55.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 20:54 5 HH:mm"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 20:54"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 20&#x2236;54"
06-01 20:54:00.985+0900 D/indicator( 2229): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 366928 Time: <font_size=34>20&#x2236;54</font_size></font>"
06-01 20:54:09.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:09.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:54:12.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:12.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:13.815+0900 F/sio_packet(23736): accept()
06-01 20:54:13.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:13.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:54:14.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:14.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:21.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:21.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:54:22.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:22.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:23.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:23.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:54:24.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:24.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:25.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:25.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:54:26.965+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:26.965+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:32.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:32.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:54:33.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:33.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:34.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:34.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 20:54:36.975+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:36.975+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:38.815+0900 F/sio_packet(23736): accept()
06-01 20:54:38.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:38.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
06-01 20:54:39.980+0900 D/indicator( 2229): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 20:54:39.980+0900 D/indicator( 2229): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
06-01 20:54:53.975+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 20:54:58.050+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: app_efl_exit(138) > app_efl_exit
06-01 20:54:58.050+0900 D/AUL     (23736): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
06-01 20:54:58.050+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 22
06-01 20:54:58.295+0900 D/APP_CORE(23736): appcore-efl.c: __after_loop(1060) > [APP 23736] PAUSE before termination
06-01 20:54:58.295+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: _ui_app_appcore_pause(603) > app_appcore_pause
06-01 20:54:58.295+0900 I/CAPI_APPFW_APPLICATION(23736): app_main.c: _ui_app_appcore_terminate(577) > app_appcore_terminate
06-01 20:54:58.445+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x65a090 
06-01 20:54:58.660+0900 W/CRASH_MANAGER(23789): worker.c: worker_job(1189) > 11237366f7468143315969
