S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 8852
Date: 2015-04-02 18:37:17+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xaf1d7008

Register Information
r0   = 0xaf1d8000, r1   = 0xaf1d7008
r2   = 0x00000800, r3   = 0xaf1d7008
r4   = 0x00000010, r5   = 0xaf1d7008
r6   = 0xaf1d8000, r7   = 0x00000000
r8   = 0x00000800, r9   = 0xaf1d7008
r10  = 0x00000200, fp   = 0x00000000
ip   = 0x00000000, sp   = 0xbebee7f8
lr   = 0xb3b76910, pc   = 0xb3b76ae4
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    459224 KB
Buffers:     26936 KB
Cached:     119488 KB
VmPeak:     143224 KB
VmSize:     118748 KB
VmLck:           0 KB
VmHWM:       15836 KB
VmRSS:       15836 KB
VmData:      62924 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24920 KB
VmPTE:          84 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 8852 TID = 8852
8852 8856 8857 8858 8859 8860 8861 8862 

Maps Information
b1813000 b1814000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b1851000 b1852000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b185a000 b1861000 r-xp /usr/lib/libfeedback.so.0.1.4
b1874000 b1875000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b187d000 b1894000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b1a1a000 b1a1f000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b3268000 b32b3000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b32bc000 b32c6000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b32cf000 b332b000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3b38000 b3b3e000 r-xp /usr/lib/libUMP.so
b3b46000 b3b59000 r-xp /usr/lib/libEGL_platform.so
b3b62000 b3c39000 r-xp /usr/lib/libMali.so
b3c44000 b3c5b000 r-xp /usr/lib/libEGL.so.1.4
b3c64000 b3c69000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c72000 b3c73000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c7c000 b3c94000 r-xp /usr/lib/libpng12.so.0.50.0
b3c9c000 b3cda000 r-xp /usr/lib/libGLESv2.so.2.0
b3ce2000 b3ce6000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3cef000 b3cf1000 r-xp /usr/lib/libdri2.so.0.0.0
b3cf9000 b3d00000 r-xp /usr/lib/libdrm.so.2.4.0
b3d09000 b3dbf000 r-xp /usr/lib/libcairo.so.2.11200.14
b3dca000 b3de0000 r-xp /usr/lib/libtts.so
b3de9000 b3df0000 r-xp /usr/lib/libtbm.so.1.0.0
b3df8000 b3dfd000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3e05000 b3e16000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3e1e000 b3e25000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3e2d000 b3e32000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3e3a000 b3e3c000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3e44000 b3e4c000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3e54000 b3e57000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3e60000 b3f0a000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3f14000 b3f1e000 r-xp /lib/libnss_files-2.13.so
b3f2c000 b3f2e000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4137000 b4158000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b4161000 b417e000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b4187000 b4255000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b426c000 b4292000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b429c000 b429e000 r-xp /usr/lib/libiniparser.so.0
b42a8000 b42ae000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b42b7000 b42bd000 r-xp /usr/lib/libappsvc.so.0.1.0
b42c6000 b42c8000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b42d1000 b42d5000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b42dd000 b42e1000 r-xp /usr/lib/libogg.so.0.7.1
b42e9000 b430b000 r-xp /usr/lib/libvorbis.so.0.4.3
b4313000 b43f7000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b440b000 b443c000 r-xp /usr/lib/libFLAC.so.8.2.0
b4445000 b4447000 r-xp /usr/lib/libXau.so.6.0.0
b444f000 b449b000 r-xp /usr/lib/libssl.so.1.0.0
b44a8000 b44d6000 r-xp /usr/lib/libidn.so.11.5.44
b44de000 b44e8000 r-xp /usr/lib/libcares.so.2.1.0
b44f0000 b4535000 r-xp /usr/lib/libsndfile.so.1.0.25
b4543000 b454a000 r-xp /usr/lib/libsensord-share.so
b4552000 b4568000 r-xp /lib/libexpat.so.1.5.2
b4576000 b4579000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b4581000 b45b5000 r-xp /usr/lib/libicule.so.51.1
b45be000 b45d1000 r-xp /usr/lib/libxcb.so.1.1.0
b45d9000 b4614000 r-xp /usr/lib/libcurl.so.4.3.0
b461d000 b4626000 r-xp /usr/lib/libethumb.so.1.7.99
b5b94000 b5c28000 r-xp /usr/lib/libstdc++.so.6.0.16
b5c3b000 b5c3d000 r-xp /usr/lib/libctxdata.so.0.0.0
b5c45000 b5c52000 r-xp /usr/lib/libremix.so.0.0.0
b5c5a000 b5c5b000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c63000 b5c7a000 r-xp /usr/lib/liblua-5.1.so
b5c83000 b5c8a000 r-xp /usr/lib/libembryo.so.1.7.99
b5c92000 b5cb5000 r-xp /usr/lib/libjpeg.so.8.0.2
b5ccd000 b5ce3000 r-xp /usr/lib/libsensor.so.1.1.0
b5cec000 b5d42000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5d4f000 b5d72000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d7b000 b5dc1000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5dca000 b5ddd000 r-xp /usr/lib/libfribidi.so.0.3.1
b5de5000 b5e35000 r-xp /usr/lib/libfreetype.so.6.8.1
b5e40000 b5e43000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5e4b000 b5e4f000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e57000 b5e5c000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e65000 b5e6f000 r-xp /usr/lib/libXext.so.6.4.0
b5e77000 b5f58000 r-xp /usr/lib/libX11.so.6.3.0
b5f63000 b5f66000 r-xp /usr/lib/libXtst.so.6.1.0
b5f6e000 b5f74000 r-xp /usr/lib/libXrender.so.1.3.0
b5f7c000 b5f81000 r-xp /usr/lib/libXrandr.so.2.2.0
b5f89000 b5f8a000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f93000 b5f9b000 r-xp /usr/lib/libXi.so.6.1.0
b5f9c000 b5f9f000 r-xp /usr/lib/libXfixes.so.3.1.0
b5fa7000 b5fa9000 r-xp /usr/lib/libXgesture.so.7.0.0
b5fb1000 b5fb3000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5fbb000 b5fbc000 r-xp /usr/lib/libXdamage.so.1.1.0
b5fc5000 b5fcb000 r-xp /usr/lib/libXcursor.so.1.0.2
b5fd4000 b5fed000 r-xp /usr/lib/libecore_con.so.1.7.99
b5ff7000 b5ffd000 r-xp /usr/lib/libecore_imf.so.1.7.99
b6005000 b600d000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6015000 b6019000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b6022000 b6038000 r-xp /usr/lib/libefreet.so.1.7.99
b6041000 b604a000 r-xp /usr/lib/libedbus.so.1.7.99
b6052000 b6137000 r-xp /usr/lib/libicuuc.so.51.1
b614c000 b628b000 r-xp /usr/lib/libicui18n.so.51.1
b629b000 b62f7000 r-xp /usr/lib/libedje.so.1.7.99
b6301000 b6312000 r-xp /usr/lib/libecore_input.so.1.7.99
b631a000 b631f000 r-xp /usr/lib/libecore_file.so.1.7.99
b6327000 b6340000 r-xp /usr/lib/libeet.so.1.7.99
b6351000 b6355000 r-xp /usr/lib/libappcore-common.so.1.1
b635e000 b642a000 r-xp /usr/lib/libevas.so.1.7.99
b644f000 b6470000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6479000 b64a8000 r-xp /usr/lib/libecore_x.so.1.7.99
b64b2000 b65e6000 r-xp /usr/lib/libelementary.so.1.7.99
b65fe000 b65ff000 r-xp /usr/lib/libjournal.so.0.1.0
b6608000 b66d3000 r-xp /usr/lib/libxml2.so.2.7.8
b66e1000 b66f1000 r-xp /lib/libresolv-2.13.so
b66f5000 b670b000 r-xp /lib/libz.so.1.2.5
b6713000 b6715000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b671d000 b6722000 r-xp /usr/lib/libffi.so.5.0.10
b672b000 b672c000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6734000 b6737000 r-xp /lib/libattr.so.1.1.0
b673f000 b68e7000 r-xp /usr/lib/libcrypto.so.1.0.0
b6907000 b6921000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b692a000 b6993000 r-xp /lib/libm-2.13.so
b699c000 b69dc000 r-xp /usr/lib/libeina.so.1.7.99
b69e5000 b69ed000 r-xp /usr/lib/libvconf.so.0.2.45
b69f5000 b69f8000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6a00000 b6a34000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a3d000 b6b11000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b1d000 b6b23000 r-xp /lib/librt-2.13.so
b6b2c000 b6b31000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6b3a000 b6b41000 r-xp /lib/libcrypt-2.13.so
b6b71000 b6b74000 r-xp /lib/libcap.so.2.21
b6b7c000 b6b7e000 r-xp /usr/lib/libiri.so
b6b86000 b6ba5000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6bad000 b6bc3000 r-xp /usr/lib/libecore.so.1.7.99
b6bd9000 b6bde000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6be7000 b6cb7000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6cb8000 b6cc6000 r-xp /usr/lib/libail.so.0.1.0
b6cce000 b6ce5000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6cee000 b6cf8000 r-xp /lib/libunwind.so.8.0.1
b6d26000 b6e41000 r-xp /lib/libc-2.13.so
b6e4f000 b6e57000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e5f000 b6e89000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e92000 b6e95000 r-xp /usr/lib/libbundle.so.0.1.22
b6e9d000 b6e9f000 r-xp /lib/libdl-2.13.so
b6ea8000 b6eab000 r-xp /usr/lib/libsmack.so.1.0.0
b6eb3000 b6f15000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f1f000 b6f31000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f3a000 b6f4e000 r-xp /lib/libpthread-2.13.so
b6f5b000 b6f5f000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f69000 b6f6b000 r-xp /usr/lib/libdlog.so.0.0.0
b6f73000 b6f7e000 r-xp /usr/lib/libaul.so.0.1.0
b6f88000 b6f8c000 r-xp /usr/lib/libsys-assert.so
b6f95000 b6fb2000 r-xp /lib/ld-2.13.so
b6fbb000 b6fc1000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6fc9000 b6ff5000 rw-p [heap]
b6ff5000 b71ef000 rw-p [heap]
bebcf000 bebf0000 rwxp [stack]
End of Maps Information

Callstack Information (PID:8852)
Call Stack Count: 1
 0: (0xb3b76ae4) [/usr/lib/libMali.so] + 0x14ae4
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
2 18:37:15.175+0900 D/RESOURCED( 2381): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-02 18:37:15.175+0900 D/RESOURCED( 2381): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-02 18:37:15.175+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-02 18:37:15.175+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-02 18:37:15.175+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-02 18:37:15.175+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-02 18:37:15.175+0900 E/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 573
04-02 18:37:15.175+0900 D/RESOURCED( 2381): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-02 18:37:15.175+0900 D/AUL     ( 2229): launch.c: app_request_to_launchpad(295) > launch request result : 8847
04-02 18:37:15.180+0900 D/STARTER ( 2229): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
04-02 18:37:15.210+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: main(461) > Application Main Function 
04-02 18:37:15.210+0900 D/LAUNCH  ( 8847): appcore-efl.c: appcore_efl_main(1569) > [task-mgr:Application:main:done]
04-02 18:37:15.220+0900 D/AUL     ( 8847): pkginfo.c: aul_app_get_appid_bypid(196) > [SECURE_LOG] appid for 8847 is org.tizen.task-mgr
04-02 18:37:15.370+0900 D/APP_CORE( 8847): appcore-efl.c: __before_loop(1012) > elm_config_preferred_engine_set : opengl_x11
04-02 18:37:15.380+0900 D/AUL     ( 8847): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 8847 is org.tizen.task-mgr
04-02 18:37:15.380+0900 D/APP_CORE( 8847): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.task-mgr/res/locale
04-02 18:37:15.380+0900 D/APP_CORE( 8847): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-02 18:37:15.380+0900 D/AUL     ( 8847): app_sock.c: __create_server_sock(135) > pg path - already exists
04-02 18:37:15.380+0900 D/LAUNCH  ( 8847): appcore-efl.c: __before_loop(1035) > [task-mgr:Platform:appcore_init:done]
04-02 18:37:15.380+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: _create_app(331) > Application Create Callback 
04-02 18:37:15.555+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: create_layout(197) > create_layout
04-02 18:37:15.560+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: create_layout(216) > Resolution: HD
04-02 18:37:15.560+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: load_data(282) > load_data
04-02 18:37:15.560+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_create(625) > recent_app_list_create
04-02 18:37:15.565+0900 D/PKGMGR_INFO( 8847): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3109) > [SECURE_LOG] where = package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
04-02 18:37:15.565+0900 D/PKGMGR_INFO( 8847): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3115) > [SECURE_LOG] query = select DISTINCT package_app_info.*, package_app_localized_info.app_locale, package_app_localized_info.app_label, package_app_localized_info.app_icon from package_app_info LEFT OUTER JOIN package_app_localized_info ON package_app_info.app_id=package_app_localized_info.app_id and package_app_localized_info.app_locale IN ('No Locale', 'en-us') LEFT OUTER JOIN package_app_app_svc ON package_app_info.app_id=package_app_app_svc.app_id LEFT OUTER JOIN package_app_app_category ON package_app_info.app_id=package_app_app_category.app_id where package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
04-02 18:37:15.570+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 12
04-02 18:37:15.570+0900 D/AUL     ( 8847): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 343
04-02 18:37:15.570+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_create(653) > USP mode is disabled.
04-02 18:37:15.575+0900 E/RUA     ( 8847): rua.c: rua_history_load_db(278) > rua_history_load_db ok. nrows : 2, ncols : 5
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_create(680) > Apps in history: 2
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_retrieve_item(444) > [org.tizen.other]
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): [/opt/usr/apps/org.tizen.other/shared/res/other.png]
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): [other]
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_create(741) > App added into the running_list : pkgid:org.tizen.other - appid:org.tizen.other
04-02 18:37:15.575+0900 E/TASK_MGR_LITE( 8847): recent_apps.c: list_retrieve_item(348) > Fail to get ai table !!
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_create(773) > HISTORY LIST: (nil) 0
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_create(774) > RUNNING LIST: 0x174398 1
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: load_data(292) > LISTs : 0x1bde00, 0x1bd480
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: load_data(304) > App list should display !!
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_init(258) > in gen list: 0x174398 (nil)
04-02 18:37:15.575+0900 D/TASK_MGR_LITE( 8847): genlist.c: recent_app_panel_create(98) > Creating task_mgr_genlist widget, noti count is : [-1]
04-02 18:37:15.595+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: genlist_item_class_create(308) > Genlist item class create
04-02 18:37:15.595+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: genlist_clear_all_class_create(323) > Genlist clear all class create
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_update(432) > genlist_update
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_clear_list(297) > genlist_clear_list
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: add_clear_btn_to_genlist(417) > add_clear_btn_to_genlist
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_add_item(165) > genlist_add_item
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_add_item(171) > Adding item: 0x1bc0e0
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid(210) > recent_apps_find_by_appid
04-02 18:37:15.615+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 12
04-02 18:37:15.615+0900 D/AUL     ( 8847): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 343
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.lockscreen (2243)
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.volume (2253)
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.menu-screen (2254)
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.other (8815)
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid_cb(201) > FOUND
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.task-mgr (8847)
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_update(449) > Are there recent apps: (nil)
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_update(450) > Adding recent apps... 0
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_item_show(192) > genlist_item_show
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_item_show(199) > ELM COUNT = 2
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: load_data(323) > load_data END. 
04-02 18:37:15.615+0900 D/LAUNCH  ( 8847): appcore-efl.c: __before_loop(1045) > [task-mgr:Application:create:done]
04-02 18:37:15.615+0900 D/APP_CORE( 8847): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-02 18:37:15.615+0900 D/APP_CORE( 8847): appcore.c: __aul_handler(423) > [APP 8847]     AUL event: AUL_START
04-02 18:37:15.615+0900 D/APP_CORE( 8847): appcore-efl.c: __do_app(470) > [APP 8847] Event: RESET State: CREATED
04-02 18:37:15.615+0900 D/APP_CORE( 8847): appcore-efl.c: __do_app(496) > [APP 8847] RESET
04-02 18:37:15.615+0900 D/LAUNCH  ( 8847): appcore-efl.c: __do_app(498) > [task-mgr:Application:reset:start]
04-02 18:37:15.615+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: _reset_app(446) > _reset_app
04-02 18:37:15.615+0900 D/LAUNCH  ( 8847): appcore-efl.c: __do_app(501) > [task-mgr:Application:reset:done]
04-02 18:37:15.615+0900 E/PKGMGR_INFO( 8847): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_appinfo(1308) > (appid == NULL) appid is NULL
04-02 18:37:15.620+0900 I/APP_CORE( 8847): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-02 18:37:15.620+0900 I/APP_CORE( 8847): appcore-efl.c: __do_app(509) > [APP 8847] Initial Launching, call the resume_cb
04-02 18:37:15.620+0900 D/APP_CORE( 8847): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-02 18:37:15.620+0900 D/APP_CORE( 8847): appcore.c: __prt_ltime(183) > [APP 8847] first idle after reset: 731 msec
04-02 18:37:15.655+0900 W/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=8847
04-02 18:37:15.710+0900 D/APP_CORE( 8847): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:3600008
04-02 18:37:15.710+0900 D/APP_CORE( 8847): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:3600008
04-02 18:37:15.710+0900 D/indicator( 2225): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2600003"
04-02 18:37:15.715+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 3600008"
04-02 18:37:15.715+0900 D/indicator( 2225): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-02 18:37:15.720+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 3600008"
04-02 18:37:15.720+0900 D/indicator( 2225): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-02 18:37:15.815+0900 D/RESOURCED( 2381): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 8847, type = 0
04-02 18:37:15.815+0900 D/AUL_AMD ( 2178): amd_launch.c: __e17_status_handler(1888) > pid(8847) status(3)
04-02 18:37:15.815+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 8847
04-02 18:37:15.815+0900 D/RESOURCED( 2381): cpu.c: cpu_foreground_state(92) > [cpu_foreground_state,92] cpu_foreground_state : pid = 8847, appname = (null)
04-02 18:37:15.815+0900 D/RESOURCED( 2381): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/cgroup.procs, value 8847
04-02 18:37:15.835+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
04-02 18:37:15.840+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
04-02 18:37:15.840+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/other.png; other 0x23d2d0
04-02 18:37:15.840+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(169) > It is already set.
04-02 18:37:15.840+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: recent_app_item_create(543) > 
04-02 18:37:15.840+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_main_create(765) > 
04-02 18:37:15.850+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
04-02 18:37:15.850+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
04-02 18:37:15.850+0900 E/TASK_MGR_LITE( 8847): genlist_item.c: del_cb(758) > Deleted
04-02 18:37:15.855+0900 D/APP_CORE( 8847): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:3600008 fully_obscured 0
04-02 18:37:15.855+0900 D/APP_CORE( 8847): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-02 18:37:15.855+0900 D/APP_CORE( 8847): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-02 18:37:15.855+0900 D/APP_CORE( 8847): appcore-efl.c: __do_app(470) > [APP 8847] Event: RESUME State: RUNNING
04-02 18:37:15.855+0900 D/LAUNCH  ( 8847): appcore-efl.c: __do_app(557) > [task-mgr:Application:resume:start]
04-02 18:37:15.855+0900 D/LAUNCH  ( 8847): appcore-efl.c: __do_app(567) > [task-mgr:Application:resume:done]
04-02 18:37:15.855+0900 D/LAUNCH  ( 8847): appcore-efl.c: __do_app(569) > [task-mgr:Application:Launching:done]
04-02 18:37:15.855+0900 D/APP_CORE( 8847): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-02 18:37:15.855+0900 E/APP_CORE( 8847): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/other.png; other 0x23d2d0
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: recent_app_item_create(543) > 
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_main_create(765) > 
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
04-02 18:37:15.860+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
04-02 18:37:15.955+0900 D/STARTER ( 2229): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
04-02 18:37:15.955+0900 W/STARTER ( 2229): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
04-02 18:37:15.955+0900 D/STARTER ( 2229): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
04-02 18:37:15.955+0900 D/STARTER ( 2229): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
04-02 18:37:15.965+0900 D/STARTER ( 2229): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
04-02 18:37:15.965+0900 I/SYSPOPUP( 2253): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
04-02 18:37:15.970+0900 I/SYSPOPUP( 2253): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
04-02 18:37:15.970+0900 D/VOLUME  ( 2253): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
04-02 18:37:15.970+0900 E/VOLUME  ( 2253): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
04-02 18:37:15.970+0900 E/VOLUME  ( 2253): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
04-02 18:37:15.970+0900 D/VOLUME  ( 2253): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
04-02 18:37:15.970+0900 D/VOLUME  ( 2253): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
04-02 18:37:15.970+0900 D/VOLUME  ( 2253): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
04-02 18:37:15.970+0900 E/VOLUME  ( 2253): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
04-02 18:37:15.970+0900 D/VOLUME  ( 2253): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
04-02 18:37:16.180+0900 D/AUL_AMD ( 2178): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.task-mgr /usr/apps/org.tizen.task-mgr/bin/task-mgr
04-02 18:37:16.180+0900 D/RUA     ( 2178): rua.c: rua_add_history(179) > rua_add_history start
04-02 18:37:16.200+0900 D/RUA     ( 2178): rua.c: rua_add_history(247) > rua_add_history ok
04-02 18:37:16.265+0900 D/APP_CORE( 2254): appcore-efl.c: __do_app(470) > [APP 2254] Event: MEM_FLUSH State: PAUSED
04-02 18:37:16.305+0900 D/EFL     ( 8847): ecore_x<8847> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=12094003 button=1
04-02 18:37:16.360+0900 D/EFL     ( 8847): ecore_x<8847> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=12094059 button=1
04-02 18:37:16.365+0900 D/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x60374c 
04-02 18:37:16.365+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
04-02 18:37:16.365+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected list 0.000000 no_shutdown: 0
04-02 18:37:16.365+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
04-02 18:37:16.365+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected layout -0.000000 no_shutdown: 0
04-02 18:37:16.365+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: clear_all_btn_clicked_cb(218) > Removing all items...
04-02 18:37:16.370+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: clear_all_btn_clicked_cb(230) > On REDWOOD target (HD) : No Animation.
04-02 18:37:16.370+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_apps_kill_all(164) > recent_apps_kill_all
04-02 18:37:16.370+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 12
04-02 18:37:16.370+0900 D/AUL     ( 8847): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 343
04-02 18:37:16.380+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: get_app_taskmanage(121) > app org.tizen.lockscreen is taskmanage: 0
04-02 18:37:16.395+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: get_app_taskmanage(121) > app org.tizen.volume is taskmanage: 0
04-02 18:37:16.405+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: get_app_taskmanage(121) > app org.tizen.menu-screen is taskmanage: 0
04-02 18:37:16.415+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: get_app_taskmanage(121) > app org.tizen.other is taskmanage: 1
04-02 18:37:16.415+0900 D/AUL     ( 8847): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : 8815
04-02 18:37:16.415+0900 D/AUL     ( 8847): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(4)
04-02 18:37:16.420+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 4
04-02 18:37:16.420+0900 D/RESOURCED( 2381): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-02 18:37:16.420+0900 D/RESOURCED( 2381): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2178
04-02 18:37:16.420+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid (null), pid 8815, type 6 
04-02 18:37:16.420+0900 E/RESOURCED( 2381): datausage-common.c: app_terminate_cb(254) > [app_terminate_cb,254] No classid to terminate!
04-02 18:37:16.420+0900 D/RESOURCED( 2381): proc-main.c: proc_remove_process_list(363) > [proc_remove_process_list,363] found_pid 8815
04-02 18:37:16.420+0900 D/AUL_AMD ( 2178): amd_request.c: __app_process_by_pid(175) > __app_process_by_pid, cmd: 4, pid: 8815, 
04-02 18:37:16.420+0900 D/AUL     ( 2178): app_sock.c: __app_send_raw_with_delay_reply(434) > pid(8815) : cmd(4)
04-02 18:37:16.420+0900 D/AUL_AMD ( 2178): amd_launch.c: _term_app(878) > term done
04-02 18:37:16.420+0900 D/AUL_AMD ( 2178): amd_launch.c: __set_reply_handler(860) > listen fd : 28, send fd : 14
04-02 18:37:16.425+0900 D/AUL_AMD ( 2178): amd_launch.c: __reply_handler(784) > listen fd(28) , send fd(14), pid(8815), cmd(4)
04-02 18:37:16.425+0900 D/APP_CORE( 8815): appcore.c: __aul_handler(443) > [APP 8815]     AUL event: AUL_TERMINATE
04-02 18:37:16.425+0900 D/AUL     ( 8847): launch.c: app_request_to_launchpad(295) > launch request result : 0
04-02 18:37:16.425+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: kill_pid(141) > terminate pid = 8815
04-02 18:37:16.425+0900 D/APP_CORE( 8815): appcore-efl.c: __do_app(470) > [APP 8815] Event: TERMINATE State: RUNNING
04-02 18:37:16.425+0900 D/APP_CORE( 8815): appcore-efl.c: __do_app(486) > [APP 8815] TERMINATE
04-02 18:37:16.425+0900 D/AUL     ( 8815): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
04-02 18:37:16.425+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 22
04-02 18:37:16.435+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: get_app_taskmanage(121) > app org.tizen.task-mgr is taskmanage: 0
04-02 18:37:16.435+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_del(189) > _recent_app_item_content_del
04-02 18:37:16.435+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_del(194) > Data exist: 0xb0b02fd0
04-02 18:37:16.435+0900 E/TASK_MGR_LITE( 8847): genlist_item.c: del_cb(758) > Deleted
04-02 18:37:16.435+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _recent_app_item_content_del(198) > Item deleted
04-02 18:37:16.435+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _clear_all_button_del(206) > _clear_all_button_del
04-02 18:37:16.440+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: _clear_all_button_del(213) > _clear_all_button_del END
04-02 18:37:16.455+0900 D/AUL     ( 8847): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
04-02 18:37:16.455+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 22
04-02 18:37:16.455+0900 D/APP_CORE( 8847): appcore-efl.c: __after_loop(1060) > [APP 8847] PAUSE before termination
04-02 18:37:16.455+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: _pause_app(453) > _pause_app
04-02 18:37:16.455+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: _terminate_app(393) > _terminate_app
04-02 18:37:16.455+0900 D/TASK_MGR_LITE( 8847): genlist.c: genlist_clear_list(297) > genlist_clear_list
04-02 18:37:16.455+0900 D/TASK_MGR_LITE( 8847): genlist.c: recent_app_panel_del_cb(84) > recent_app_panel_del_cb
04-02 18:37:16.455+0900 D/TASK_MGR_LITE( 8847): genlist_item.c: genlist_item_class_destroy(340) > Genlist class free
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): task-mgr-lite.c: delete_layout(272) > delete_layout
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: recent_app_list_destroy(791) > recent_app_list_destroy
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_destroy(475) > START list_destroy
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_destroy(487) > FREE ALL list_destroy
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_destroy(493) > FREING: list_destroy org.tizen.other
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_unretrieve_item(295) > FREING 0x1bc0e0 org.tizen.other 's Item(list_type_default_s) in list_unretrieve_item
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_unretrieve_item(333) > list_unretrieve_item END 
04-02 18:37:16.460+0900 D/TASK_MGR_LITE( 8847): recent_apps.c: list_destroy(500) > END list_destroy
04-02 18:37:16.480+0900 E/APP_CORE( 8847): appcore.c: appcore_flush_memory(583) > Appcore not initialized
04-02 18:37:16.480+0900 W/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=8815
04-02 18:37:16.495+0900 D/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_wininfo_del(141) > [PROCESSMGR] delete anr_trigger_timer!
04-02 18:37:16.495+0900 D/indicator( 2225): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 3600008"
04-02 18:37:16.495+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
04-02 18:37:16.495+0900 D/indicator( 2225): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-02 18:37:16.500+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
04-02 18:37:16.500+0900 D/indicator( 2225): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-02 18:37:16.610+0900 I/CAPI_APPFW_APPLICATION( 8815): app_main.c: _ui_app_appcore_terminate(577) > app_appcore_terminate
04-02 18:37:16.610+0900 D/RESOURCED( 2381): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2254, type = 0
04-02 18:37:16.610+0900 D/AUL_AMD ( 2178): amd_launch.c: __e17_status_handler(1888) > pid(2254) status(3)
04-02 18:37:16.610+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 2254
04-02 18:37:16.610+0900 D/RESOURCED( 2381): cpu.c: cpu_foreground_state(92) > [cpu_foreground_state,92] cpu_foreground_state : pid = 2254, appname = (null)
04-02 18:37:16.610+0900 D/RESOURCED( 2381): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/cgroup.procs, value 2254
04-02 18:37:16.780+0900 D/indicator( 2225): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2600003"
04-02 18:37:16.780+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1c00003"
04-02 18:37:16.780+0900 D/indicator( 2225): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-02 18:37:16.780+0900 W/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2254
04-02 18:37:16.780+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1c00003"
04-02 18:37:16.780+0900 D/indicator( 2225): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-02 18:37:16.845+0900 D/APP_CORE( 2254): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 0
04-02 18:37:16.845+0900 D/APP_CORE( 2254): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active 0
04-02 18:37:16.845+0900 D/APP_CORE( 2254): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-02 18:37:16.845+0900 D/APP_CORE( 2254): appcore-efl.c: __do_app(470) > [APP 2254] Event: RESUME State: PAUSED
04-02 18:37:16.845+0900 D/LAUNCH  ( 2254): appcore-efl.c: __do_app(557) > [menu-screen:Application:resume:start]
04-02 18:37:16.845+0900 D/APP_CORE( 2254): appcore-efl.c: __do_app(559) > [APP 2254] RESUME
04-02 18:37:16.845+0900 I/CAPI_APPFW_APPLICATION( 2254): app_main.c: app_appcore_resume(216) > app_appcore_resume
04-02 18:37:16.845+0900 D/MENU_SCREEN( 2254): menu_screen.c: _resume_cb(547) > START RESUME
04-02 18:37:16.845+0900 D/MENU_SCREEN( 2254): page_scroller.c: page_scroller_focus(1133) > Focus set scroller(0xb71ea858), page:0, item:other
04-02 18:37:16.845+0900 D/LAUNCH  ( 2254): appcore-efl.c: __do_app(567) > [menu-screen:Application:resume:done]
04-02 18:37:16.845+0900 D/LAUNCH  ( 2254): appcore-efl.c: __do_app(569) > [menu-screen:Application:Launching:done]
04-02 18:37:16.845+0900 D/APP_CORE( 2254): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-02 18:37:16.845+0900 E/APP_CORE( 2254): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-02 18:37:17.080+0900 I/AUL_PAD ( 2213): sigchild.h: __launchpad_sig_child(142) > dead_pid = 8847 pgid = 8847
04-02 18:37:17.080+0900 I/AUL_PAD ( 2213): sigchild.h: __sigchild_action(123) > dead_pid(8847)
04-02 18:37:17.080+0900 D/AUL_PAD ( 2213): sigchild.h: __send_app_dead_signal(81) > send dead signal done
04-02 18:37:17.080+0900 I/AUL_PAD ( 2213): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
04-02 18:37:17.080+0900 I/AUL_PAD ( 2213): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
04-02 18:37:17.100+0900 D/STARTER ( 2229): lock-daemon-lite.c: lockd_app_dead_cb_lite(998) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:998:D] app dead cb call! (pid : 8847)
04-02 18:37:17.100+0900 D/STARTER ( 2229): menu_daemon.c: menu_daemon_check_dead_signal(621) > [menu_daemon_check_dead_signal:621] Process 8847 is termianted
04-02 18:37:17.100+0900 D/STARTER ( 2229): menu_daemon.c: menu_daemon_check_dead_signal(646) > [menu_daemon_check_dead_signal:646] Unknown process, ignore it (dead pid 8847, home pid 2254, taskmgr pid -1)
04-02 18:37:17.105+0900 I/AUL_AMD ( 2178): amd_main.c: __app_dead_handler(257) > __app_dead_handler, pid: 8847
04-02 18:37:17.105+0900 D/AUL_AMD ( 2178): amd_key.c: _unregister_key_event(156) > ===key stack===
04-02 18:37:17.105+0900 D/AUL     ( 2178): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-02 18:37:17.105+0900 E/AUL     ( 2178): simple_util.c: __trm_app_info_send_socket(264) > access
04-02 18:37:17.340+0900 D/EFL     ( 2254): ecore_x<2254> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=12095038 button=1
04-02 18:37:17.345+0900 D/MENU_SCREEN( 2254): mouse.c: _down_cb(103) > Mouse down (96,309)
04-02 18:37:17.345+0900 D/MENU_SCREEN( 2254): item_event.c: _item_down_cb(63) > ITEM: mouse down event callback is invoked for 0xb16853c0
04-02 18:37:17.395+0900 D/EFL     ( 2254): ecore_x<2254> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=12095094 button=1
04-02 18:37:17.400+0900 D/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x60006b 
04-02 18:37:17.400+0900 D/MENU_SCREEN( 2254): mouse.c: _up_cb(120) > Mouse up (102,310)
04-02 18:37:17.400+0900 D/MENU_SCREEN( 2254): item.c: _focus_clicked_cb(668) > ITEM: mouse up event callback is invoked for 0xb16853c0
04-02 18:37:17.400+0900 D/MENU_SCREEN( 2254): layout.c: layout_enable_block(116) > Enable layout blocker
04-02 18:37:17.400+0900 D/AUL     ( 2254): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.other
04-02 18:37:17.400+0900 D/AUL     ( 2254): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(1)
04-02 18:37:17.400+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(491) > __request_handler: 1
04-02 18:37:17.400+0900 D/AUL_AMD ( 2178): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-02 18:37:17.400+0900 D/AUL_AMD ( 2178): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : org.tizen.menu-screen
04-02 18:37:17.410+0900 D/AUL_AMD ( 2178): amd_launch.c: _start_app(1591) > win(c00002) ecore_x_pointer_grab(1)
04-02 18:37:17.410+0900 E/AUL_AMD ( 2178): amd_launch.c: invoke_dbus_method_sync(1177) > dbus_connection_send error(org.freedesktop.DBus.Error.ServiceUnknown:The name org.tizen.system.coord was not provided by any .service files)
04-02 18:37:17.410+0900 D/AUL_AMD ( 2178): amd_launch.c: _start_app(1675) > org.tizen.system.coord.rotation-Degree : -74
04-02 18:37:17.410+0900 D/AUL_AMD ( 2178): amd_launch.c: _start_app(1770) > process_pool: false
04-02 18:37:17.410+0900 D/AUL_AMD ( 2178): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-02 18:37:17.410+0900 D/AUL_AMD ( 2178): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-02 18:37:17.410+0900 D/AUL_AMD ( 2178): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-02 18:37:17.410+0900 D/AUL     ( 2178): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(1)
04-02 18:37:17.410+0900 D/AUL_PAD ( 2213): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-02 18:37:17.410+0900 D/AUL_PAD ( 2213): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-02 18:37:17.410+0900 D/AUL_PAD ( 2213): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 8852 /opt/usr/apps/org.tizen.other/bin/other
04-02 18:37:17.410+0900 D/AUL_PAD ( 8852): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-02 18:37:17.410+0900 D/AUL_PAD ( 2213): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-02 18:37:17.415+0900 D/AUL_PAD ( 8852): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-02 18:37:17.415+0900 D/AUL_PAD ( 8852): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-02 18:37:17.415+0900 I/AUL_PAD ( 2213): sigchild.h: __launchpad_sig_child(142) > dead_pid = 8815 pgid = 8815
04-02 18:37:17.415+0900 I/AUL_PAD ( 2213): sigchild.h: __sigchild_action(123) > dead_pid(8815)
04-02 18:37:17.415+0900 D/AUL_PAD ( 2213): sigchild.h: __send_app_dead_signal(81) > send dead signal done
04-02 18:37:17.415+0900 I/AUL_PAD ( 2213): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
04-02 18:37:17.415+0900 I/AUL_PAD ( 2213): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
04-02 18:37:17.415+0900 D/AUL_PAD ( 2213): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-02 18:37:17.415+0900 D/STARTER ( 2229): lock-daemon-lite.c: lockd_app_dead_cb_lite(998) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:998:D] app dead cb call! (pid : 8815)
04-02 18:37:17.415+0900 D/STARTER ( 2229): menu_daemon.c: menu_daemon_check_dead_signal(621) > [menu_daemon_check_dead_signal:621] Process 8815 is termianted
04-02 18:37:17.415+0900 D/STARTER ( 2229): menu_daemon.c: menu_daemon_check_dead_signal(646) > [menu_daemon_check_dead_signal:646] Unknown process, ignore it (dead pid 8815, home pid 2254, taskmgr pid -1)
04-02 18:37:17.440+0900 D/AUL_PAD ( 8852): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-02 18:37:17.440+0900 D/AUL_PAD ( 8852): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-02 18:37:17.440+0900 D/AUL_PAD ( 8852): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-02 18:37:17.440+0900 D/AUL_PAD ( 8852): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-02 18:37:17.440+0900 D/AUL_PAD ( 8852): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_APPID__##
04-02 18:37:17.440+0900 D/LAUNCH  ( 8852): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-02 18:37:17.475+0900 I/CAPI_APPFW_APPLICATION( 8852): app_main.c: ui_app_main(697) > app_efl_main
04-02 18:37:17.475+0900 D/LAUNCH  ( 8852): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-02 18:37:17.515+0900 D/AUL_PAD ( 2213): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-02 18:37:17.515+0900 D/AUL_PAD ( 2213): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-02 18:37:17.515+0900 D/AUL_PAD ( 2213): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-02 18:37:17.515+0900 D/AUL     ( 2178): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-02 18:37:17.515+0900 E/AUL     ( 2178): simple_util.c: __trm_app_info_send_socket(264) > access
04-02 18:37:17.515+0900 D/AUL_AMD ( 2178): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 8852
04-02 18:37:17.515+0900 D/AUL_AMD ( 2178): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-02 18:37:17.515+0900 D/AUL     ( 2254): launch.c: app_request_to_launchpad(295) > launch request result : 8852
04-02 18:37:17.515+0900 D/MENU_SCREEN( 2254): item.c: item_launch(1003) > Launch app's ret : [8852]
04-02 18:37:17.515+0900 D/LAUNCH  ( 2254): item.c: item_launch(1005) > [org.tizen.other:Menuscreen:launch:done]
04-02 18:37:17.515+0900 D/MENU_SCREEN( 2254): item_event.c: _item_up_cb(85) > ITEM: mouse up event callback is invoked for 0xb16853c0
04-02 18:37:17.515+0900 D/RESOURCED( 2381): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-02 18:37:17.515+0900 D/RESOURCED( 2381): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2178
04-02 18:37:17.515+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 8852, type 4 
04-02 18:37:17.520+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 8852
04-02 18:37:17.520+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-02 18:37:17.520+0900 D/RESOURCED( 2381): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 8852
04-02 18:37:17.520+0900 D/RESOURCED( 2381): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 8852
04-02 18:37:17.520+0900 D/RESOURCED( 2381): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-02 18:37:17.520+0900 D/RESOURCED( 2381): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-02 18:37:17.520+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-02 18:37:17.520+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-02 18:37:17.520+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-02 18:37:17.520+0900 D/RESOURCED( 2381): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-02 18:37:17.520+0900 E/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 582
04-02 18:37:17.520+0900 D/RESOURCED( 2381): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-02 18:37:17.525+0900 I/AUL_AMD ( 2178): amd_main.c: __app_dead_handler(257) > __app_dead_handler, pid: 8815
04-02 18:37:17.525+0900 D/AUL_AMD ( 2178): amd_key.c: _unregister_key_event(156) > ===key stack===
04-02 18:37:17.525+0900 D/AUL     ( 2178): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-02 18:37:17.525+0900 E/AUL     ( 2178): simple_util.c: __trm_app_info_send_socket(264) > access
04-02 18:37:17.525+0900 D/APP_CORE( 8852): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-02 18:37:17.535+0900 D/AUL     ( 8852): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 8852 is org.tizen.other
04-02 18:37:17.535+0900 D/APP_CORE( 8852): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-02 18:37:17.535+0900 D/APP_CORE( 8852): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-02 18:37:17.535+0900 D/AUL     ( 8852): app_sock.c: __create_server_sock(135) > pg path - already exists
04-02 18:37:17.535+0900 D/LAUNCH  ( 8852): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-02 18:37:17.535+0900 I/CAPI_APPFW_APPLICATION( 8852): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-02 18:37:17.700+0900 W/PROCESSMGR( 2102): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=8852
04-02 18:37:17.710+0900 D/indicator( 2225): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-02 18:37:17.710+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
04-02 18:37:17.710+0900 D/indicator( 2225): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-02 18:37:17.710+0900 D/indicator( 2225): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
04-02 18:37:17.710+0900 D/indicator( 2225): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-02 18:37:17.715+0900 F/socket.io( 8852): thread_start
04-02 18:37:17.715+0900 F/socket.io( 8852): finish 0
04-02 18:37:17.715+0900 D/LAUNCH  ( 8852): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-02 18:37:17.715+0900 D/APP_CORE( 8852): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-02 18:37:17.715+0900 D/APP_CORE( 8852): appcore.c: __aul_handler(423) > [APP 8852]     AUL event: AUL_START
04-02 18:37:17.715+0900 D/APP_CORE( 8852): appcore-efl.c: __do_app(470) > [APP 8852] Event: RESET State: CREATED
04-02 18:37:17.715+0900 D/APP_CORE( 8852): appcore-efl.c: __do_app(496) > [APP 8852] RESET
04-02 18:37:17.715+0900 D/LAUNCH  ( 8852): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-02 18:37:17.715+0900 I/CAPI_APPFW_APPLICATION( 8852): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-02 18:37:17.715+0900 D/APP_SVC ( 8852): appsvc.c: __set_bundle(161) > __set_bundle
04-02 18:37:17.715+0900 D/LAUNCH  ( 8852): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-02 18:37:17.725+0900 I/APP_CORE( 8852): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-02 18:37:17.725+0900 I/APP_CORE( 8852): appcore-efl.c: __do_app(509) > [APP 8852] Initial Launching, call the resume_cb
04-02 18:37:17.725+0900 I/CAPI_APPFW_APPLICATION( 8852): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-02 18:37:17.725+0900 D/APP_CORE( 8852): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : org.tizen.menu-screen
04-02 18:37:17.725+0900 D/APP_CORE( 8852): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600003
04-02 18:37:17.725+0900 D/APP_CORE( 8852): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600003
04-02 18:37:17.795+0900 E/socket.io( 8852): 566: Connected.
04-02 18:37:17.795+0900 E/socket.io( 8852): 554: On handshake, sid
04-02 18:37:17.795+0900 E/socket.io( 8852): 651: Received Message type(connect)
04-02 18:37:17.795+0900 E/socket.io( 8852): 489: On Connected
04-02 18:37:17.795+0900 F/sio_packet( 8852): accept()
04-02 18:37:17.795+0900 E/socket.io( 8852): 743: encoded paylod length: 13
04-02 18:37:17.800+0900 E/socket.io( 8852): 800: ping exit, con is expired? 0, ec: Operation canceled
04-02 18:37:17.865+0900 E/socket.io( 8852): 669: Received Message type(Event)
04-02 18:37:17.865+0900 F/get_binary( 8852): in get binary_message()...
04-02 18:37:17.865+0900 D/CAPI_MEDIA_IMAGE_UTIL( 8852): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-02 18:37:17.865+0900 E/socket.io( 8852): 669: Received Message type(Event)
04-02 18:37:17.870+0900 D/APP_CORE( 2254): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-02 18:37:17.870+0900 D/APP_CORE( 2254): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-02 18:37:17.870+0900 D/APP_CORE( 2254): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-02 18:37:17.870+0900 D/APP_CORE( 2254): appcore-efl.c: __do_app(470) > [APP 2254] Event: PAUSE State: RUNNING
04-02 18:37:17.870+0900 D/APP_CORE( 2254): appcore-efl.c: __do_app(538) > [APP 2254] PAUSE
04-02 18:37:17.870+0900 I/CAPI_APPFW_APPLICATION( 2254): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-02 18:37:17.870+0900 D/MENU_SCREEN( 2254): menu_screen.c: _pause_cb(538) > Pause start
04-02 18:37:17.870+0900 D/APP_CORE( 2254): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-02 18:37:17.870+0900 E/APP_CORE( 2254): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-02 18:37:17.875+0900 D/RESOURCED( 2381): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2254, type = 2
04-02 18:37:17.875+0900 D/RESOURCED( 2381): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 8852, type = 0
04-02 18:37:17.875+0900 D/AUL_AMD ( 2178): amd_launch.c: __e17_status_handler(1888) > pid(8852) status(3)
04-02 18:37:17.875+0900 D/RESOURCED( 2381): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 8852
04-02 18:37:17.880+0900 I/RESOURCED( 2381): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 8852, oom : 200
04-02 18:37:17.880+0900 E/RESOURCED( 2381): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-02 18:37:17.885+0900 D/APP_CORE( 8852): appcore.c: __prt_ltime(183) > [APP 8852] first idle after reset: 488 msec
04-02 18:37:17.885+0900 D/APP_CORE( 8852): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 0
04-02 18:37:17.890+0900 D/APP_CORE( 8852): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-02 18:37:17.890+0900 D/APP_CORE( 8852): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-02 18:37:17.890+0900 D/APP_CORE( 8852): appcore-efl.c: __do_app(470) > [APP 8852] Event: RESUME State: RUNNING
04-02 18:37:17.890+0900 D/LAUNCH  ( 8852): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-02 18:37:17.890+0900 D/LAUNCH  ( 8852): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-02 18:37:17.890+0900 D/LAUNCH  ( 8852): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-02 18:37:17.890+0900 D/APP_CORE( 8852): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-02 18:37:17.890+0900 E/APP_CORE( 8852): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-02 18:37:17.920+0900 E/socket.io( 8852): 669: Received Message type(Event)
04-02 18:37:17.920+0900 F/get_binary( 8852): in get binary_message()...
04-02 18:37:17.930+0900 D/CAPI_MEDIA_IMAGE_UTIL( 8852): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 18:37:17.930+0900 E/socket.io( 8852): 669: Received Message type(Event)
04-02 18:37:17.970+0900 E/socket.io( 8852): 669: Received Message type(Event)
04-02 18:37:17.970+0900 F/get_binary( 8852): in get binary_message()...
04-02 18:37:17.975+0900 D/CAPI_MEDIA_IMAGE_UTIL( 8852): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-02 18:37:17.980+0900 E/socket.io( 8852): 669: Received Message type(Event)
04-02 18:37:18.005+0900 W/CRASH_MANAGER( 8789): worker.c: worker_job(1189) > 11088526f7468142796743
