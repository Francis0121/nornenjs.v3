S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 4358
Date: 2015-04-01 23:42:13+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xaf1f7328

Register Information
r0   = 0xae663100, r1   = 0xaf1f7308
r2   = 0x00000800, r3   = 0xaf1f7328
r4   = 0x000000d0, r5   = 0xaf1f7308
r6   = 0xae663000, r7   = 0x0000018c
r8   = 0x00000800, r9   = 0xaf1f7008
r10  = 0x00000200, fp   = 0x000000c0
ip   = 0x00000000, sp   = 0xbe885800
lr   = 0xb3b23918, pc   = 0xb3b23ae4
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    412112 KB
Buffers:     44132 KB
Cached:     118256 KB
VmPeak:     145348 KB
VmSize:     119900 KB
VmLck:           0 KB
VmHWM:       16668 KB
VmRSS:       16668 KB
VmData:      64076 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24920 KB
VmPTE:          84 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 4358 TID = 4358
4358 4362 4363 4364 4365 4366 4367 4368 

Maps Information
b17c0000 b17c1000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17fe000 b17ff000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b1807000 b180e000 r-xp /usr/lib/libfeedback.so.0.1.4
b1821000 b1822000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b182a000 b1841000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b19c7000 b19cc000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b3215000 b3260000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b3269000 b3273000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b327c000 b32d8000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3ae5000 b3aeb000 r-xp /usr/lib/libUMP.so
b3af3000 b3b06000 r-xp /usr/lib/libEGL_platform.so
b3b0f000 b3be6000 r-xp /usr/lib/libMali.so
b3bf1000 b3c08000 r-xp /usr/lib/libEGL.so.1.4
b3c11000 b3c16000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c1f000 b3c20000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c29000 b3c41000 r-xp /usr/lib/libpng12.so.0.50.0
b3c49000 b3c87000 r-xp /usr/lib/libGLESv2.so.2.0
b3c8f000 b3c93000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c9c000 b3c9e000 r-xp /usr/lib/libdri2.so.0.0.0
b3ca6000 b3cad000 r-xp /usr/lib/libdrm.so.2.4.0
b3cb6000 b3d6c000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d77000 b3d8d000 r-xp /usr/lib/libtts.so
b3d96000 b3d9d000 r-xp /usr/lib/libtbm.so.1.0.0
b3da5000 b3daa000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3db2000 b3dc3000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3dcb000 b3dd2000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3dda000 b3ddf000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3de7000 b3de9000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3df1000 b3df9000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3e01000 b3e04000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3e0d000 b3eb7000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3ec1000 b3ecb000 r-xp /lib/libnss_files-2.13.so
b3ed9000 b3edb000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b40e4000 b4105000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b410e000 b412b000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b4134000 b4202000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4219000 b423f000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4249000 b424b000 r-xp /usr/lib/libiniparser.so.0
b4255000 b425b000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b4264000 b426a000 r-xp /usr/lib/libappsvc.so.0.1.0
b4273000 b4275000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b427e000 b4282000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b428a000 b428e000 r-xp /usr/lib/libogg.so.0.7.1
b4296000 b42b8000 r-xp /usr/lib/libvorbis.so.0.4.3
b42c0000 b43a4000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b43b8000 b43e9000 r-xp /usr/lib/libFLAC.so.8.2.0
b43f2000 b43f4000 r-xp /usr/lib/libXau.so.6.0.0
b43fc000 b4448000 r-xp /usr/lib/libssl.so.1.0.0
b4455000 b4483000 r-xp /usr/lib/libidn.so.11.5.44
b448b000 b4495000 r-xp /usr/lib/libcares.so.2.1.0
b449d000 b44e2000 r-xp /usr/lib/libsndfile.so.1.0.25
b44f0000 b44f7000 r-xp /usr/lib/libsensord-share.so
b44ff000 b4515000 r-xp /lib/libexpat.so.1.5.2
b4523000 b4526000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b452e000 b4562000 r-xp /usr/lib/libicule.so.51.1
b456b000 b457e000 r-xp /usr/lib/libxcb.so.1.1.0
b4586000 b45c1000 r-xp /usr/lib/libcurl.so.4.3.0
b45ca000 b45d3000 r-xp /usr/lib/libethumb.so.1.7.99
b5b41000 b5bd5000 r-xp /usr/lib/libstdc++.so.6.0.16
b5be8000 b5bea000 r-xp /usr/lib/libctxdata.so.0.0.0
b5bf2000 b5bff000 r-xp /usr/lib/libremix.so.0.0.0
b5c07000 b5c08000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c10000 b5c27000 r-xp /usr/lib/liblua-5.1.so
b5c30000 b5c37000 r-xp /usr/lib/libembryo.so.1.7.99
b5c3f000 b5c62000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c7a000 b5c90000 r-xp /usr/lib/libsensor.so.1.1.0
b5c99000 b5cef000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5cfc000 b5d1f000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d28000 b5d6e000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d77000 b5d8a000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d92000 b5de2000 r-xp /usr/lib/libfreetype.so.6.8.1
b5ded000 b5df0000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5df8000 b5dfc000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e04000 b5e09000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e12000 b5e1c000 r-xp /usr/lib/libXext.so.6.4.0
b5e24000 b5f05000 r-xp /usr/lib/libX11.so.6.3.0
b5f10000 b5f13000 r-xp /usr/lib/libXtst.so.6.1.0
b5f1b000 b5f21000 r-xp /usr/lib/libXrender.so.1.3.0
b5f29000 b5f2e000 r-xp /usr/lib/libXrandr.so.2.2.0
b5f36000 b5f37000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f40000 b5f48000 r-xp /usr/lib/libXi.so.6.1.0
b5f49000 b5f4c000 r-xp /usr/lib/libXfixes.so.3.1.0
b5f54000 b5f56000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f5e000 b5f60000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f68000 b5f69000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f72000 b5f78000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f81000 b5f9a000 r-xp /usr/lib/libecore_con.so.1.7.99
b5fa4000 b5faa000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5fb2000 b5fba000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5fc2000 b5fc6000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5fcf000 b5fe5000 r-xp /usr/lib/libefreet.so.1.7.99
b5fee000 b5ff7000 r-xp /usr/lib/libedbus.so.1.7.99
b5fff000 b60e4000 r-xp /usr/lib/libicuuc.so.51.1
b60f9000 b6238000 r-xp /usr/lib/libicui18n.so.51.1
b6248000 b62a4000 r-xp /usr/lib/libedje.so.1.7.99
b62ae000 b62bf000 r-xp /usr/lib/libecore_input.so.1.7.99
b62c7000 b62cc000 r-xp /usr/lib/libecore_file.so.1.7.99
b62d4000 b62ed000 r-xp /usr/lib/libeet.so.1.7.99
b62fe000 b6302000 r-xp /usr/lib/libappcore-common.so.1.1
b630b000 b63d7000 r-xp /usr/lib/libevas.so.1.7.99
b63fc000 b641d000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6426000 b6455000 r-xp /usr/lib/libecore_x.so.1.7.99
b645f000 b6593000 r-xp /usr/lib/libelementary.so.1.7.99
b65ab000 b65ac000 r-xp /usr/lib/libjournal.so.0.1.0
b65b5000 b6680000 r-xp /usr/lib/libxml2.so.2.7.8
b668e000 b669e000 r-xp /lib/libresolv-2.13.so
b66a2000 b66b8000 r-xp /lib/libz.so.1.2.5
b66c0000 b66c2000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b66ca000 b66cf000 r-xp /usr/lib/libffi.so.5.0.10
b66d8000 b66d9000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b66e1000 b66e4000 r-xp /lib/libattr.so.1.1.0
b66ec000 b6894000 r-xp /usr/lib/libcrypto.so.1.0.0
b68b4000 b68ce000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b68d7000 b6940000 r-xp /lib/libm-2.13.so
b6949000 b6989000 r-xp /usr/lib/libeina.so.1.7.99
b6992000 b699a000 r-xp /usr/lib/libvconf.so.0.2.45
b69a2000 b69a5000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b69ad000 b69e1000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b69ea000 b6abe000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6aca000 b6ad0000 r-xp /lib/librt-2.13.so
b6ad9000 b6ade000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6ae7000 b6aee000 r-xp /lib/libcrypt-2.13.so
b6b1e000 b6b21000 r-xp /lib/libcap.so.2.21
b6b29000 b6b2b000 r-xp /usr/lib/libiri.so
b6b33000 b6b52000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b5a000 b6b70000 r-xp /usr/lib/libecore.so.1.7.99
b6b86000 b6b8b000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b94000 b6c64000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c65000 b6c73000 r-xp /usr/lib/libail.so.0.1.0
b6c7b000 b6c92000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c9b000 b6ca5000 r-xp /lib/libunwind.so.8.0.1
b6cd3000 b6dee000 r-xp /lib/libc-2.13.so
b6dfc000 b6e04000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e0c000 b6e36000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e3f000 b6e42000 r-xp /usr/lib/libbundle.so.0.1.22
b6e4a000 b6e4c000 r-xp /lib/libdl-2.13.so
b6e55000 b6e58000 r-xp /usr/lib/libsmack.so.1.0.0
b6e60000 b6ec2000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6ecc000 b6ede000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6ee7000 b6efb000 r-xp /lib/libpthread-2.13.so
b6f08000 b6f0c000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f16000 b6f18000 r-xp /usr/lib/libdlog.so.0.0.0
b6f20000 b6f2b000 r-xp /usr/lib/libaul.so.0.1.0
b6f35000 b6f39000 r-xp /usr/lib/libsys-assert.so
b6f42000 b6f5f000 r-xp /lib/ld-2.13.so
b6f68000 b6f6e000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f76000 b6fa2000 rw-p [heap]
b6fa2000 b71bd000 rw-p [heap]
be866000 be887000 rwxp [stack]
End of Maps Information

Callstack Information (PID:4358)
Call Stack Count: 1
 0: (0xb3b23ae4) [/usr/lib/libMali.so] + 0x14ae4
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
ALL[0m:3164] [org.tizen.other]
04-01 23:42:04.250+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(841) > [SECURE_LOG] PLUGIN_POST_PROCESS : [dynamicbox] result=[0]
04-01 23:42:04.250+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(830) > [SECURE_LOG] PLUGIN_PRE_PROCESS : [account] result=[-1]
04-01 23:42:04.250+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(841) > [SECURE_LOG] PLUGIN_POST_PROCESS : [account] result=[-1]
04-01 23:42:04.250+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.255+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(830) > [SECURE_LOG] PLUGIN_PRE_PROCESS : [privileges] result=[-1]
04-01 23:42:04.260+0900 I/PRIVACY-MANAGER-CLIENT( 4294): SocketClient.cpp: SocketClient(37) > Client created
04-01 23:42:04.265+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __run_tag_parser(513) > Plugin = privileges, appid = org.tizen.other, result=0
04-01 23:42:04.265+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __check_enabled_plugin(410) > [org.tizen.other] flag = 0x10 done[action=0]!! 
04-01 23:42:04.265+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(841) > [SECURE_LOG] PLUGIN_POST_PROCESS : [privileges] result=[-1]
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.265+0900 E/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_each_plugin(827) > (plugin_info->lib_handle == NULL) dlopen is failed lib_path
04-01 23:42:04.280+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_plugin_db(801) > [SECURE_LOG] pkgid [org.tizen.other] plugin[0x10] deleted
04-01 23:42:04.290+0900 D/PKGMGR_PLUGIN( 4294): pkgmgr_parser_plugin.c: __process_plugin_db(817) > [SECURE_LOG] pkgid [org.tizen.other] plugin[0x10] inserted
04-01 23:42:04.295+0900 D/AIL     ( 4294): ail_filter.c: ail_filter_count_appinfo(300) > No filter exists. All records are retreived
04-01 23:42:04.305+0900 E/AIL     ( 4294): ail_db.c: db_exec(244) > Cannot execute because table app_info already exists
04-01 23:42:04.305+0900 E/AIL     ( 4294): ail_db.c: db_exec(245) > query - CREATE TABLE app_info (package TEXT PRIMARY KEY, exec TEXT DEFAULT 'No Exec', name TEXT DEFAULT 'No Name', type TEXT DEFAULT 'Application', icon TEXT DEFAULT 'No Icon', categories TEXT, version TEXT, mimetype TEXT, x_slp_service TEXT, x_slp_packagetype TEXT, x_slp_packagecategories TEXT, x_slp_packageid TEXT, x_slp_uri TEXT, x_slp_svc TEXT, x_slp_exe_path TEXT, x_slp_appid TEXT, x_slp_pkgid TEXT, x_slp_domain TEXT, x_slp_submodemainid TEXT, x_slp_installedstorage TEXT, x_slp_baselayoutwidth INTEGER DEFAULT 0, x_slp_installedtime INTEGER DEFAULT 0, nodisplay INTEGER DEFAULT 0, x_slp_taskmanage INTEGER DEFAULT 1, x_slp_multiple INTEGER DEFAULT 0, x_slp_removable INTEGER DEFAULT 1, x_slp_ishorizontalscale INTEGER DEFAULT 0, x_slp_enabled INTEGER DEFAULT 1, x_slp_submode INTEGER DEFAULT 0, desktop TEXT UNIQUE NOT NULL);
04-01 23:42:04.305+0900 E/AIL     ( 4294): ail_desktop.c: _create_table(1195) > (ret != AIL_ERROR_OK) -> _create_table() return
04-01 23:42:04.305+0900 D/AIL     ( 4294): ail_desktop.c: ail_desktop_add(1590) > Cannot create a table. Maybe there is already a table.
04-01 23:42:04.305+0900 D/AIL     ( 4294): ail_desktop.c: _read_desktop_info(962) > Read (org.tizen.other).
04-01 23:42:04.315+0900 D/AIL     ( 4294): ail_desktop.c: _insert_desktop_info(1309) > Add (org.tizen.other).
04-01 23:42:04.330+0900 D/AIL     ( 4294): ail_desktop.c: _send_db_done_noti(1513) > Noti : create:org.tizen.other
04-01 23:42:04.330+0900 D/PKGMGR_PARSER( 4294): pkgmgr_parser.c: pkgmgr_parser_parse_manifest_for_installation(2186) > Creating desktop file Success
04-01 23:42:04.330+0900 D/PKGMGR_PARSER( 4294): pkgmgr_parser.c: pkgmgr_parser_parse_manifest_for_installation(2191) > Free Done
04-01 23:42:04.330+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_package_install(2164) > manifest parsing done successfully.
04-01 23:42:04.340+0900 D/rpm-installer( 4294): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[60]
04-01 23:42:04.345+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(427) > Transaction Begin
04-01 23:42:04.345+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2282): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/PKGMGR  ( 2284): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/DATA_PROVIDER_MASTER( 2284): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(419) > event_cb is called
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2210): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2405): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2287): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/QUICKPANEL( 2287): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2254): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-01 23:42:04.345+0900 D/MENU_SCREEN( 2254): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:60] for org.tizen.other
04-01 23:42:04.345+0900 D/MENU_SCREEN( 2254): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 60
04-01 23:42:04.345+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.345+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.345+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 1 11
04-01 23:42:04.345+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 2 11
04-01 23:42:04.350+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 10 1
04-01 23:42:04.350+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 4 2
04-01 23:42:04.350+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 5 2
04-01 23:42:04.350+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 6 2
04-01 23:42:04.365+0900 E/PKGMGR_CERT( 4294): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(571) > Transaction Commit and End
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/res]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_make_directory(1490) > mkdir failed. appdir=[/usr/apps/org.tizen.other/shared], errno=[2][No such file or directory]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/opt/usr/apps/org.tizen.other/shared/data]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared/res]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/tizen-manifest.xml]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/author-signature.xml]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/signature1.xml]
04-01 23:42:04.370+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/share/packages/org.tizen.other.xml]
04-01 23:42:04.370+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_make_directory_for_ext(1353) > Directory dose not exist. path: /opt/usr/apps/org.tizen.other/shared/data, errno: 2 (No such file or directory)
04-01 23:42:04.450+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_register_package(64) > [smack] app_install(org.tizen.other), result=[0]
04-01 23:42:04.450+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other, 5, _), result=[0]
04-01 23:42:04.450+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared, 5, _), result=[0]
04-01 23:42:04.450+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/res, 5, _), result=[0]
04-01 23:42:04.450+0900 E/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_get_smack_label_access(627) > Error in getting smack ACCESS label failed. result[-1] (path:[/opt/usr/apps/org.tizen.other/shared/data]))
04-01 23:42:04.455+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_get_group_id(1770) > encoding done, len=[28]
04-01 23:42:04.455+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_apply_smack(1878) > groupid = [QVI9+j1mlN94KHCZf1TZSIDuLJk=] for shared/trusted.
04-01 23:42:04.515+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/trusted, 1, QVI9+j1mlN94KHCZf1TZSIDuLJk=), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/bin, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/lib, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/tizen-manifest.xml, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/author-signature.xml, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/signature1.xml, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/share/packages/org.tizen.other.xml, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other, 5, _), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-01 23:42:04.520+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/shared, 5, _), result=[0]
04-01 23:42:04.530+0900 D/rpm-installer( 4294): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/internet]
04-01 23:42:04.540+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-01 23:42:04.545+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-01 23:42:04.630+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-01 23:42:04.630+0900 D/rpm-installer( 4294): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-01 23:42:04.630+0900 D/rpm-installer( 4294): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/systemsettings]
04-01 23:42:04.645+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-01 23:42:04.650+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-01 23:42:04.730+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-01 23:42:04.730+0900 D/rpm-installer( 4294): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-01 23:42:04.745+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-01 23:42:04.745+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-01 23:42:04.825+0900 D/rpm-installer( 4294): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-01 23:42:04.825+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_package_install(2194) > permission applying done successfully.
04-01 23:42:04.830+0900 D/PRIVILEGE_INFO( 4294): privilege_info.c: privilege_manager_verify_privilege_list(625) > privilege_info_compare_privilege_level called
04-01 23:42:04.830+0900 D/PRIVILEGE_INFO( 4294): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/internet
04-01 23:42:04.830+0900 D/PRIVILEGE_INFO( 4294): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/systemsettings
04-01 23:42:04.830+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_verify_privilege_list(672) > privilege_manager_verify_privilege_list(PRVMGR_PACKAGE_TYPE_CORE) is ok.
04-01 23:42:04.830+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_package_install(2202) > _coretpk_installer_verify_privilege_list done.
04-01 23:42:04.830+0900 D/rpm-installer( 4294): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[100]
04-01 23:42:04.830+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_package_install(2224) > install status is [2].
04-01 23:42:04.835+0900 D/rpm-installer( 4294): coretpk-installer.c: __post_install_for_mmc(311) > Installed storage is internal.
04-01 23:42:04.835+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_package_install(2231) > _coretpk_installer_package_install is done.
04-01 23:42:04.835+0900 D/rpm-installer( 4294): rpm-vconf-intf.c: _ri_broadcast_status_notification(196) > pkgid=[org.tizen.other], key=[end], val=[ok]
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.835+0900 D/DATA_PROVIDER_MASTER( 2284): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2210): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.835+0900 D/DATA_PROVIDER_MASTER( 2284): pkgmgr.c: end_cb(409) > [SECURE_LOG] [org.tizen.other] ok
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(419) > event_cb is called
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2282): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): pkgmgr.c: __operation_callback(419) > event_cb is called
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 4292): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 2254): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.835+0900 D/MENU_SCREEN( 2254): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:100] for org.tizen.other
04-01 23:42:04.835+0900 D/MENU_SCREEN( 2254): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 100
04-01 23:42:04.835+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.835+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.835+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.835+0900 D/PKGMGR  ( 2254): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.835+0900 D/MENU_SCREEN( 2254): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [end:ok] for org.tizen.other
04-01 23:42:04.835+0900 D/MENU_SCREEN( 2254): pkgmgr.c: _end(520) > Package(org.tizen.other) : key(install) - val(ok)
04-01 23:42:04.840+0900 D/rpm-installer( 4294): coretpk-installer.c: _coretpk_installer_prepare_package_install(2695) > success
04-01 23:42:04.840+0900 D/rpm-installer( 4294): rpm-appcore-intf.c: main(224) > sync() start
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.840+0900 D/PKGMGR  ( 2175): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [install] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.840+0900 D/PKGMGR  ( 2175): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.840+0900 D/AUL_AMD ( 2175): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(538) > [SECURE_LOG] pkgid(org.tizen.other), key(end), value(ok)
04-01 23:42:04.840+0900 D/AUL_AMD ( 2175): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(559) > [SECURE_LOG] op(install), value(ok)
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.840+0900 D/PKGMGR  ( 2405): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / install_percent / 100
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 23:42:04.840+0900 D/QUICKPANEL( 2287): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:100
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145 / coretpk / org.tizen.other / end / ok
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_-1209733145] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 23:42:04.840+0900 D/QUICKPANEL( 2287): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:end val:ok
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.840+0900 D/PKGMGR  ( 2287): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.890+0900 D/AUL_AMD ( 2175): amd_appinfo.c: __app_info_insert_handler(185) > __app_info_insert_handler
04-01 23:42:04.890+0900 D/AUL_AMD ( 2175): amd_appinfo.c: __app_info_insert_handler(388) > [SECURE_LOG] appinfo file:org.tizen.other, comp:ui, type:rpm
04-01 23:42:04.890+0900 D/PKGMGR  ( 2175): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.890+0900 D/PKGMGR  ( 2175): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:04.890+0900 D/MENU_SCREEN( 2254): layout.c: layout_create_package(200) > package org.tizen.other is installed directly
04-01 23:42:04.900+0900 D/MENU_SCREEN( 2254): item.c: item_update(594) > Access to file [/opt/usr/apps/org.tizen.other/shared/res/other.png], size[57662]
04-01 23:42:04.910+0900 D/BADGE   ( 2254): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
04-01 23:42:04.935+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 23:42:04.935+0900 D/PKGMGR  ( 2254): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 23:42:05.270+0900 D/rpm-installer( 4294): rpm-appcore-intf.c: main(226) > sync() end
04-01 23:42:05.270+0900 D/rpm-installer( 4294): rpm-appcore-intf.c: main(245) > ------------------------------------------------
04-01 23:42:05.270+0900 D/rpm-installer( 4294): rpm-appcore-intf.c: main(246) >  [END] rpm-installer: result=[0]
04-01 23:42:05.270+0900 D/rpm-installer( 4294): rpm-appcore-intf.c: main(247) > ------------------------------------------------
04-01 23:42:05.290+0900 D/PKGMGR_SERVER( 4284): pkgmgr-server.c: sighandler(326) > child exit [4294]
04-01 23:42:05.290+0900 D/PKGMGR_SERVER( 4284): pkgmgr-server.c: sighandler(341) > child NORMAL exit [4294]
04-01 23:42:06.105+0900 D/PKGMGR_SERVER( 4284): pkgmgr-server.c: exit_server(724) > exit_server Start
04-01 23:42:06.105+0900 D/PKGMGR_SERVER( 4284): pkgmgr-server.c: main(1516) > Quit main loop.
04-01 23:42:06.105+0900 D/PKGMGR_SERVER( 4284): pkgmgr-server.c: main(1524) > package manager server terminated.
04-01 23:42:12.290+0900 D/AUL_AMD ( 2175): amd_request.c: __request_handler(491) > __request_handler: 0
04-01 23:42:12.295+0900 D/AUL_AMD ( 2175): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-01 23:42:12.330+0900 D/AUL     ( 2175): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 4355, pid = 4357
04-01 23:42:12.365+0900 D/AUL_AMD ( 2175): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-01 23:42:12.390+0900 D/AUL_AMD ( 2175): amd_launch.c: _start_app(1770) > process_pool: false
04-01 23:42:12.390+0900 D/AUL_AMD ( 2175): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-01 23:42:12.390+0900 D/AUL_AMD ( 2175): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-01 23:42:12.390+0900 D/AUL_AMD ( 2175): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-01 23:42:12.390+0900 D/AUL     ( 2175): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-01 23:42:12.390+0900 D/AUL_PAD ( 2216): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-01 23:42:12.390+0900 D/AUL_PAD ( 2216): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-01 23:42:12.400+0900 D/AUL_PAD ( 2216): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 4358 /opt/usr/apps/org.tizen.other/bin/other
04-01 23:42:12.400+0900 D/AUL_PAD ( 2216): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-01 23:42:12.400+0900 D/AUL_PAD ( 4358): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-01 23:42:12.400+0900 D/AUL_PAD ( 4358): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-01 23:42:12.405+0900 D/AUL_PAD ( 4358): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-01 23:42:12.425+0900 D/AUL_PAD ( 4358): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-01 23:42:12.425+0900 D/AUL_PAD ( 4358): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-01 23:42:12.425+0900 D/AUL_PAD ( 4358): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-01 23:42:12.425+0900 D/AUL_PAD ( 4358): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-01 23:42:12.425+0900 D/LAUNCH  ( 4358): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-01 23:42:12.455+0900 I/CAPI_APPFW_APPLICATION( 4358): app_main.c: ui_app_main(697) > app_efl_main
04-01 23:42:12.455+0900 D/LAUNCH  ( 4358): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-01 23:42:12.490+0900 D/APP_CORE( 4358): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-01 23:42:12.500+0900 D/AUL     ( 2175): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-01 23:42:12.500+0900 E/AUL     ( 2175): simple_util.c: __trm_app_info_send_socket(264) > access
04-01 23:42:12.500+0900 D/RESOURCED( 2371): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-01 23:42:12.500+0900 D/RESOURCED( 2371): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2175
04-01 23:42:12.500+0900 D/RESOURCED( 2371): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 4358, type 4 
04-01 23:42:12.500+0900 D/AUL_AMD ( 2175): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 4358
04-01 23:42:12.500+0900 D/AUL_AMD ( 2175): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-01 23:42:12.500+0900 D/RESOURCED( 2371): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 4358
04-01 23:42:12.500+0900 D/RESOURCED( 2371): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-01 23:42:12.500+0900 D/RESOURCED( 2371): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 4358
04-01 23:42:12.500+0900 D/RESOURCED( 2371): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 4358
04-01 23:42:12.500+0900 D/RESOURCED( 2371): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-01 23:42:12.500+0900 D/RESOURCED( 2371): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-01 23:42:12.500+0900 D/RESOURCED( 2371): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-01 23:42:12.500+0900 D/RESOURCED( 2371): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-01 23:42:12.500+0900 D/RESOURCED( 2371): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-01 23:42:12.500+0900 D/RESOURCED( 2371): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-01 23:42:12.500+0900 E/RESOURCED( 2371): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 536
04-01 23:42:12.500+0900 D/RESOURCED( 2371): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-01 23:42:12.505+0900 D/AUL     ( 4358): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 4358 is org.tizen.other
04-01 23:42:12.505+0900 D/AUL_PAD ( 2216): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-01 23:42:12.505+0900 D/APP_CORE( 4358): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-01 23:42:12.505+0900 D/AUL_PAD ( 2216): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-01 23:42:12.505+0900 D/AUL_PAD ( 2216): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-01 23:42:12.505+0900 D/APP_CORE( 4358): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-01 23:42:12.505+0900 D/AUL     ( 4358): app_sock.c: __create_server_sock(135) > pg path - already exists
04-01 23:42:12.505+0900 D/LAUNCH  ( 4358): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-01 23:42:12.505+0900 I/CAPI_APPFW_APPLICATION( 4358): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-01 23:42:12.710+0900 F/socket.io( 4358): thread_start
04-01 23:42:12.710+0900 F/socket.io( 4358): finish 0
04-01 23:42:12.710+0900 D/LAUNCH  ( 4358): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-01 23:42:12.710+0900 D/APP_CORE( 4358): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-01 23:42:12.710+0900 D/APP_CORE( 4358): appcore.c: __aul_handler(423) > [APP 4358]     AUL event: AUL_START
04-01 23:42:12.710+0900 D/APP_CORE( 4358): appcore-efl.c: __do_app(470) > [APP 4358] Event: RESET State: CREATED
04-01 23:42:12.710+0900 D/APP_CORE( 4358): appcore-efl.c: __do_app(496) > [APP 4358] RESET
04-01 23:42:12.710+0900 D/LAUNCH  ( 4358): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-01 23:42:12.710+0900 I/CAPI_APPFW_APPLICATION( 4358): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-01 23:42:12.710+0900 D/APP_SVC ( 4358): appsvc.c: __set_bundle(161) > __set_bundle
04-01 23:42:12.710+0900 D/LAUNCH  ( 4358): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-01 23:42:12.720+0900 I/APP_CORE( 4358): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-01 23:42:12.720+0900 I/APP_CORE( 4358): appcore-efl.c: __do_app(509) > [APP 4358] Initial Launching, call the resume_cb
04-01 23:42:12.720+0900 I/CAPI_APPFW_APPLICATION( 4358): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-01 23:42:12.720+0900 D/APP_CORE( 4358): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-01 23:42:12.720+0900 D/APP_CORE( 4358): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:3800003
04-01 23:42:12.720+0900 D/APP_CORE( 4358): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:3800003
04-01 23:42:12.880+0900 D/APP_CORE( 4358): appcore.c: __prt_ltime(183) > [APP 4358] first idle after reset: 592 msec
04-01 23:42:12.890+0900 D/RESOURCED( 2371): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2254, type = 2
04-01 23:42:12.905+0900 D/APP_CORE( 4358): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:3800003 fully_obscured 1
04-01 23:42:12.905+0900 D/APP_CORE( 4358): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active -1
04-01 23:42:12.905+0900 D/APP_CORE( 4358): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-01 23:42:12.905+0900 D/APP_CORE( 4358): appcore-efl.c: __do_app(470) > [APP 4358] Event: PAUSE State: RUNNING
04-01 23:42:12.905+0900 D/APP_CORE( 4358): appcore-efl.c: __do_app(538) > [APP 4358] PAUSE
04-01 23:42:12.905+0900 I/CAPI_APPFW_APPLICATION( 4358): app_main.c: _ui_app_appcore_pause(603) > app_appcore_pause
04-01 23:42:12.905+0900 D/APP_CORE( 4358): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-01 23:42:12.905+0900 E/APP_CORE( 4358): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-01 23:42:12.905+0900 E/socket.io( 4358): 566: Connected.
04-01 23:42:12.905+0900 E/socket.io( 4358): 554: On handshake, sid
04-01 23:42:12.910+0900 E/socket.io( 4358): 651: Received Message type(connect)
04-01 23:42:12.910+0900 E/socket.io( 4358): 489: On Connected
04-01 23:42:12.925+0900 F/sio_packet( 4358): accept()
04-01 23:42:12.925+0900 E/socket.io( 4358): 743: encoded paylod length: 13
04-01 23:42:12.925+0900 E/socket.io( 4358): 800: ping exit, con is expired? 0, ec: Operation canceled
04-01 23:42:13.000+0900 E/socket.io( 4358): 669: Received Message type(Event)
04-01 23:42:13.000+0900 F/get_binary( 4358): in get binary_message()...
04-01 23:42:13.000+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4358): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-01 23:42:13.000+0900 E/socket.io( 4358): 669: Received Message type(Event)
04-01 23:42:13.060+0900 E/socket.io( 4358): 669: Received Message type(Event)
04-01 23:42:13.060+0900 F/get_binary( 4358): in get binary_message()...
04-01 23:42:13.070+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4358): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 23:42:13.070+0900 E/socket.io( 4358): 669: Received Message type(Event)
04-01 23:42:13.120+0900 E/socket.io( 4358): 669: Received Message type(Event)
04-01 23:42:13.120+0900 F/get_binary( 4358): in get binary_message()...
04-01 23:42:13.125+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4358): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 23:42:13.130+0900 E/socket.io( 4358): 669: Received Message type(Event)
04-01 23:42:13.355+0900 W/CRASH_MANAGER( 4369): worker.c: worker_job(1189) > 11043586f7468142789933
