S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 4589
Date: 2015-04-01 20:52:08+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xaf3a5428

Register Information
r0   = 0xae6a4100, r1   = 0xaf3a5408
r2   = 0x00000800, r3   = 0xaf3a5428
r4   = 0x00000110, r5   = 0xaf3a5408
r6   = 0xae6a4000, r7   = 0x00000050
r8   = 0x00000800, r9   = 0xaf3a5008
r10  = 0x00000200, fp   = 0x00000020
ip   = 0x00000000, sp   = 0xbe807800
lr   = 0xb3ba1918, pc   = 0xb3ba1ae4
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    450972 KB
Buffers:     18812 KB
Cached:     128952 KB
VmPeak:     145768 KB
VmSize:     120576 KB
VmLck:           0 KB
VmHWM:       16764 KB
VmRSS:       16764 KB
VmData:      64672 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24996 KB
VmPTE:          88 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 4589 TID = 4589
4589 4593 4594 4595 4596 4597 4598 4599 

Maps Information
b183e000 b183f000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b187c000 b187d000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b1885000 b188c000 r-xp /usr/lib/libfeedback.so.0.1.4
b189f000 b18a0000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b18a8000 b18bf000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b1a45000 b1a4a000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b3293000 b32de000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b32e7000 b32f1000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b32fa000 b3356000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3b63000 b3b69000 r-xp /usr/lib/libUMP.so
b3b71000 b3b84000 r-xp /usr/lib/libEGL_platform.so
b3b8d000 b3c64000 r-xp /usr/lib/libMali.so
b3c6f000 b3c86000 r-xp /usr/lib/libEGL.so.1.4
b3c8f000 b3c94000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c9d000 b3c9e000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3ca7000 b3cbf000 r-xp /usr/lib/libpng12.so.0.50.0
b3cc7000 b3d05000 r-xp /usr/lib/libGLESv2.so.2.0
b3d0d000 b3d11000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3d1a000 b3d1c000 r-xp /usr/lib/libdri2.so.0.0.0
b3d24000 b3d2b000 r-xp /usr/lib/libdrm.so.2.4.0
b3d34000 b3dea000 r-xp /usr/lib/libcairo.so.2.11200.14
b3df5000 b3e0b000 r-xp /usr/lib/libtts.so
b3e14000 b3e1b000 r-xp /usr/lib/libtbm.so.1.0.0
b3e23000 b3e28000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3e30000 b3e41000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3e49000 b3e50000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3e58000 b3e5d000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3e65000 b3e67000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3e6f000 b3e77000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3e7f000 b3e82000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3e8b000 b3f48000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3f53000 b3f5d000 r-xp /lib/libnss_files-2.13.so
b3f6b000 b3f6d000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4176000 b4197000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b41a0000 b41bd000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b41c6000 b4294000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b42ab000 b42d1000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b42db000 b42dd000 r-xp /usr/lib/libiniparser.so.0
b42e7000 b42ed000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b42f6000 b42fc000 r-xp /usr/lib/libappsvc.so.0.1.0
b4305000 b4307000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4310000 b4314000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b431c000 b4320000 r-xp /usr/lib/libogg.so.0.7.1
b4328000 b434a000 r-xp /usr/lib/libvorbis.so.0.4.3
b4352000 b4436000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b444a000 b447b000 r-xp /usr/lib/libFLAC.so.8.2.0
b4484000 b4486000 r-xp /usr/lib/libXau.so.6.0.0
b448e000 b44da000 r-xp /usr/lib/libssl.so.1.0.0
b44e7000 b4515000 r-xp /usr/lib/libidn.so.11.5.44
b451d000 b4527000 r-xp /usr/lib/libcares.so.2.1.0
b452f000 b4574000 r-xp /usr/lib/libsndfile.so.1.0.25
b4582000 b4589000 r-xp /usr/lib/libsensord-share.so
b4591000 b45a7000 r-xp /lib/libexpat.so.1.5.2
b45b5000 b45b8000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b45c0000 b45f4000 r-xp /usr/lib/libicule.so.51.1
b45fd000 b4610000 r-xp /usr/lib/libxcb.so.1.1.0
b4618000 b4653000 r-xp /usr/lib/libcurl.so.4.3.0
b465c000 b4665000 r-xp /usr/lib/libethumb.so.1.7.99
b5bd3000 b5c67000 r-xp /usr/lib/libstdc++.so.6.0.16
b5c7a000 b5c7c000 r-xp /usr/lib/libctxdata.so.0.0.0
b5c84000 b5c91000 r-xp /usr/lib/libremix.so.0.0.0
b5c99000 b5c9a000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5ca2000 b5cb9000 r-xp /usr/lib/liblua-5.1.so
b5cc2000 b5cc9000 r-xp /usr/lib/libembryo.so.1.7.99
b5cd1000 b5cf4000 r-xp /usr/lib/libjpeg.so.8.0.2
b5d0c000 b5d22000 r-xp /usr/lib/libsensor.so.1.1.0
b5d2b000 b5d81000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5d8e000 b5db1000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5dba000 b5e00000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5e09000 b5e1c000 r-xp /usr/lib/libfribidi.so.0.3.1
b5e24000 b5e74000 r-xp /usr/lib/libfreetype.so.6.8.1
b5e7f000 b5e82000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5e8a000 b5e8e000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e96000 b5e9b000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5ea4000 b5eae000 r-xp /usr/lib/libXext.so.6.4.0
b5eb6000 b5f97000 r-xp /usr/lib/libX11.so.6.3.0
b5fa2000 b5fa5000 r-xp /usr/lib/libXtst.so.6.1.0
b5fad000 b5fb3000 r-xp /usr/lib/libXrender.so.1.3.0
b5fbb000 b5fc0000 r-xp /usr/lib/libXrandr.so.2.2.0
b5fc8000 b5fc9000 r-xp /usr/lib/libXinerama.so.1.0.0
b5fd2000 b5fda000 r-xp /usr/lib/libXi.so.6.1.0
b5fdb000 b5fde000 r-xp /usr/lib/libXfixes.so.3.1.0
b5fe6000 b5fe8000 r-xp /usr/lib/libXgesture.so.7.0.0
b5ff0000 b5ff2000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5ffa000 b5ffb000 r-xp /usr/lib/libXdamage.so.1.1.0
b6004000 b600a000 r-xp /usr/lib/libXcursor.so.1.0.2
b6013000 b602c000 r-xp /usr/lib/libecore_con.so.1.7.99
b6036000 b603c000 r-xp /usr/lib/libecore_imf.so.1.7.99
b6044000 b604c000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6054000 b6058000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b6061000 b6077000 r-xp /usr/lib/libefreet.so.1.7.99
b6080000 b6089000 r-xp /usr/lib/libedbus.so.1.7.99
b6091000 b6176000 r-xp /usr/lib/libicuuc.so.51.1
b618b000 b62ca000 r-xp /usr/lib/libicui18n.so.51.1
b62da000 b6336000 r-xp /usr/lib/libedje.so.1.7.99
b6340000 b6351000 r-xp /usr/lib/libecore_input.so.1.7.99
b6359000 b635e000 r-xp /usr/lib/libecore_file.so.1.7.99
b6366000 b637f000 r-xp /usr/lib/libeet.so.1.7.99
b6390000 b6394000 r-xp /usr/lib/libappcore-common.so.1.1
b639d000 b6469000 r-xp /usr/lib/libevas.so.1.7.99
b648e000 b64af000 r-xp /usr/lib/libecore_evas.so.1.7.99
b64b8000 b64e7000 r-xp /usr/lib/libecore_x.so.1.7.99
b64f1000 b6625000 r-xp /usr/lib/libelementary.so.1.7.99
b663d000 b663e000 r-xp /usr/lib/libjournal.so.0.1.0
b6647000 b6712000 r-xp /usr/lib/libxml2.so.2.7.8
b6720000 b6730000 r-xp /lib/libresolv-2.13.so
b6734000 b674a000 r-xp /lib/libz.so.1.2.5
b6752000 b6754000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b675c000 b6761000 r-xp /usr/lib/libffi.so.5.0.10
b676a000 b676b000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6773000 b6776000 r-xp /lib/libattr.so.1.1.0
b677e000 b6926000 r-xp /usr/lib/libcrypto.so.1.0.0
b6946000 b6960000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b6969000 b69d2000 r-xp /lib/libm-2.13.so
b69db000 b6a1b000 r-xp /usr/lib/libeina.so.1.7.99
b6a24000 b6a2c000 r-xp /usr/lib/libvconf.so.0.2.45
b6a34000 b6a37000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6a3f000 b6a73000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a7c000 b6b50000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b5c000 b6b62000 r-xp /lib/librt-2.13.so
b6b6b000 b6b70000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6b79000 b6b80000 r-xp /lib/libcrypt-2.13.so
b6bb0000 b6bb3000 r-xp /lib/libcap.so.2.21
b6bbb000 b6bbd000 r-xp /usr/lib/libiri.so
b6bc5000 b6be4000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6bec000 b6c02000 r-xp /usr/lib/libecore.so.1.7.99
b6c18000 b6c1d000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6c26000 b6cf6000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6cf7000 b6d05000 r-xp /usr/lib/libail.so.0.1.0
b6d0d000 b6d24000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6d2d000 b6d37000 r-xp /lib/libunwind.so.8.0.1
b6d65000 b6e80000 r-xp /lib/libc-2.13.so
b6e8e000 b6e96000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e9e000 b6ec8000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6ed1000 b6ed4000 r-xp /usr/lib/libbundle.so.0.1.22
b6edc000 b6ede000 r-xp /lib/libdl-2.13.so
b6ee7000 b6eea000 r-xp /usr/lib/libsmack.so.1.0.0
b6ef2000 b6f54000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f5e000 b6f70000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f79000 b6f8d000 r-xp /lib/libpthread-2.13.so
b6f9a000 b6f9e000 r-xp /usr/lib/libappcore-efl.so.1.1
b6fa8000 b6faa000 r-xp /usr/lib/libdlog.so.0.0.0
b6fb2000 b6fbd000 r-xp /usr/lib/libaul.so.0.1.0
b6fc7000 b6fcb000 r-xp /usr/lib/libsys-assert.so
b6fd4000 b6ff1000 r-xp /lib/ld-2.13.so
b6ffa000 b7000000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b7008000 b7034000 rw-p [heap]
b7034000 b72e2000 rw-p [heap]
be7e8000 be809000 rwxp [stack]
End of Maps Information

Callstack Information (PID:4589)
Call Stack Count: 1
 0: (0xb3ba1ae4) [/usr/lib/libMali.so] + 0x14ae4
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
ath(org.tizen.other, /opt/usr/apps/org.tizen.other, 5, _), result=[0]
04-01 20:51:59.310+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared, 5, _), result=[0]
04-01 20:51:59.310+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/res, 5, _), result=[0]
04-01 20:51:59.310+0900 E/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_get_smack_label_access(627) > Error in getting smack ACCESS label failed. result[-1] (path:[/opt/usr/apps/org.tizen.other/shared/data]))
04-01 20:51:59.315+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_get_group_id(1770) > encoding done, len=[28]
04-01 20:51:59.315+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_apply_smack(1878) > groupid = [QVI9+j1mlN94KHCZf1TZSIDuLJk=] for shared/trusted.
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/trusted, 1, QVI9+j1mlN94KHCZf1TZSIDuLJk=), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/bin, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/lib, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/tizen-manifest.xml, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/author-signature.xml, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/signature1.xml, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/share/packages/org.tizen.other.xml, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other, 5, _), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-01 20:51:59.375+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-01 20:51:59.380+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/shared, 5, _), result=[0]
04-01 20:51:59.380+0900 D/rpm-installer( 4528): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/internet]
04-01 20:51:59.395+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-01 20:51:59.395+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-01 20:51:59.475+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-01 20:51:59.475+0900 D/rpm-installer( 4528): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-01 20:51:59.475+0900 D/rpm-installer( 4528): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/systemsettings]
04-01 20:51:59.485+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-01 20:51:59.490+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-01 20:51:59.565+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-01 20:51:59.565+0900 D/rpm-installer( 4528): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-01 20:51:59.580+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-01 20:51:59.580+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-01 20:51:59.655+0900 D/rpm-installer( 4528): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-01 20:51:59.655+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_package_install(2194) > permission applying done successfully.
04-01 20:51:59.655+0900 D/PRIVILEGE_INFO( 4528): privilege_info.c: privilege_manager_verify_privilege_list(625) > privilege_info_compare_privilege_level called
04-01 20:51:59.655+0900 D/PRIVILEGE_INFO( 4528): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/internet
04-01 20:51:59.655+0900 D/PRIVILEGE_INFO( 4528): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/systemsettings
04-01 20:51:59.660+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_verify_privilege_list(672) > privilege_manager_verify_privilege_list(PRVMGR_PACKAGE_TYPE_CORE) is ok.
04-01 20:51:59.660+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_package_install(2202) > _coretpk_installer_verify_privilege_list done.
04-01 20:51:59.660+0900 D/rpm-installer( 4528): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[100]
04-01 20:51:59.660+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_package_install(2224) > install status is [2].
04-01 20:51:59.660+0900 D/rpm-installer( 4528): coretpk-installer.c: __post_install_for_mmc(311) > Installed storage is internal.
04-01 20:51:59.660+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_package_install(2231) > _coretpk_installer_package_install is done.
04-01 20:51:59.660+0900 D/rpm-installer( 4528): rpm-vconf-intf.c: _ri_broadcast_status_notification(196) > pkgid=[org.tizen.other], key=[end], val=[ok]
04-01 20:51:59.660+0900 D/rpm-installer( 4528): coretpk-installer.c: _coretpk_installer_prepare_package_install(2695) > success
04-01 20:51:59.660+0900 D/rpm-installer( 4528): rpm-appcore-intf.c: main(224) > sync() start
04-01 20:51:59.660+0900 D/PKGMGR  ( 2230): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2230): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/PKGMGR  ( 2230): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 2230): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.660+0900 D/PKGMGR  ( 2281): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2281): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/DATA_PROVIDER_MASTER( 2281): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2281): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 2281): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.660+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.660+0900 D/PKGMGR  ( 2239): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2239): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/MENU_SCREEN( 2239): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:100] for org.tizen.other
04-01 20:51:59.660+0900 D/MENU_SCREEN( 2239): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2239): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 2239): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): pkgmgr.c: __operation_callback(419) > event_cb is called
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 4526): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.660+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2284): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/QUICKPANEL( 2284): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.660+0900 D/PKGMGR  ( 2402): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / install_percent / 100
04-01 20:51:59.660+0900 D/PKGMGR  ( 2402): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-01 20:51:59.660+0900 D/PKGMGR  ( 2402): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.660+0900 D/PKGMGR  ( 2402): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.665+0900 D/PKGMGR  ( 2230): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2230): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/PKGMGR  ( 2230): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.665+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2230): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.665+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/PKGMGR  ( 2281): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.665+0900 D/PKGMGR  ( 2281): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.665+0900 D/DATA_PROVIDER_MASTER( 2281): pkgmgr.c: end_cb(409) > [SECURE_LOG] [org.tizen.other] ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2239): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2239): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/PKGMGR  ( 2180): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [install] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/MENU_SCREEN( 2239): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [end:ok] for org.tizen.other
04-01 20:51:59.665+0900 D/PKGMGR  ( 2180): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/MENU_SCREEN( 2239): pkgmgr.c: _end(520) > Package(org.tizen.other) : key(install) - val(ok)
04-01 20:51:59.665+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(538) > [SECURE_LOG] pkgid(org.tizen.other), key(end), value(ok)
04-01 20:51:59.665+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(559) > [SECURE_LOG] op(install), value(ok)
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-01 20:51:59.665+0900 D/PKGMGR  ( 2402): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2402): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/PKGMGR  ( 2402): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.665+0900 D/PKGMGR  ( 2402): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): pkgmgr.c: __operation_callback(419) > event_cb is called
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.665+0900 D/PKGMGR  ( 4526): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.665+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790 / coretpk / org.tizen.other / end / ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2284): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_1470014790] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-01 20:51:59.665+0900 D/QUICKPANEL( 2284): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:end val:ok
04-01 20:51:59.665+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.665+0900 D/PKGMGR  ( 2284): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.715+0900 D/PKGMGR  ( 2281): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.715+0900 D/PKGMGR  ( 2281): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.725+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __app_info_insert_handler(185) > __app_info_insert_handler
04-01 20:51:59.725+0900 D/AUL_AMD ( 2180): amd_appinfo.c: __app_info_insert_handler(388) > [SECURE_LOG] appinfo file:org.tizen.other, comp:ui, type:rpm
04-01 20:51:59.725+0900 D/PKGMGR  ( 2180): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:51:59.725+0900 D/PKGMGR  ( 2180): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:51:59.735+0900 D/MENU_SCREEN( 2239): layout.c: layout_create_package(200) > package org.tizen.other is installed directly
04-01 20:51:59.925+0900 D/indicator( 2222): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-01 20:51:59.925+0900 D/indicator( 2222): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
04-01 20:52:00.050+0900 D/MENU_SCREEN( 2239): item.c: item_update(594) > Access to file [/opt/usr/apps/org.tizen.other/shared/res/other.png], size[57662]
04-01 20:52:00.065+0900 D/BADGE   ( 2239): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
04-01 20:52:00.080+0900 D/rpm-installer( 4528): rpm-appcore-intf.c: main(226) > sync() end
04-01 20:52:00.080+0900 D/rpm-installer( 4528): rpm-appcore-intf.c: main(245) > ------------------------------------------------
04-01 20:52:00.080+0900 D/rpm-installer( 4528): rpm-appcore-intf.c: main(246) >  [END] rpm-installer: result=[0]
04-01 20:52:00.080+0900 D/rpm-installer( 4528): rpm-appcore-intf.c: main(247) > ------------------------------------------------
04-01 20:52:00.100+0900 D/PKGMGR_SERVER( 4518): pkgmgr-server.c: sighandler(326) > child exit [4528]
04-01 20:52:00.100+0900 D/PKGMGR_SERVER( 4518): pkgmgr-server.c: sighandler(341) > child NORMAL exit [4528]
04-01 20:52:00.115+0900 D/PKGMGR  ( 2239): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-01 20:52:00.115+0900 D/PKGMGR  ( 2239): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-01 20:52:00.765+0900 D/indicator( 2222): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
04-01 20:52:00.770+0900 D/indicator( 2222): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is h:mm"
04-01 20:52:00.770+0900 D/indicator( 2222): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
04-01 20:52:00.775+0900 D/indicator( 2222): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 8:52 4 h:mm"
04-01 20:52:00.775+0900 D/indicator( 2222): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 8:52"
04-01 20:52:00.775+0900 D/indicator( 2222): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 8&#x2236;52"
04-01 20:52:00.775+0900 D/indicator( 2222): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 367440 Time: <font_size=34>8&#x2236;52</font_size> <font_size=33>PM</font_size></font>"
04-01 20:52:00.910+0900 D/PKGMGR_SERVER( 4518): pkgmgr-server.c: exit_server(724) > exit_server Start
04-01 20:52:00.910+0900 D/PKGMGR_SERVER( 4518): pkgmgr-server.c: main(1516) > Quit main loop.
04-01 20:52:00.910+0900 D/PKGMGR_SERVER( 4518): pkgmgr-server.c: main(1524) > package manager server terminated.
04-01 20:52:04.930+0900 D/indicator( 2222): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-01 20:52:04.930+0900 D/indicator( 2222): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
04-01 20:52:07.100+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(491) > __request_handler: 0
04-01 20:52:07.105+0900 D/AUL_AMD ( 2180): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-01 20:52:07.140+0900 D/AUL     ( 2180): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 4586, pid = 4588
04-01 20:52:07.165+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-01 20:52:07.170+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1770) > process_pool: false
04-01 20:52:07.170+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-01 20:52:07.170+0900 D/AUL_AMD ( 2180): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-01 20:52:07.170+0900 D/AUL_AMD ( 2180): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-01 20:52:07.175+0900 D/AUL     ( 2180): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-01 20:52:07.175+0900 D/AUL_PAD ( 2220): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-01 20:52:07.175+0900 D/AUL_PAD ( 2220): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-01 20:52:07.175+0900 D/AUL_PAD ( 2220): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 4589 /opt/usr/apps/org.tizen.other/bin/other
04-01 20:52:07.175+0900 D/AUL_PAD ( 4589): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-01 20:52:07.175+0900 D/AUL_PAD ( 2220): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-01 20:52:07.175+0900 D/AUL_PAD ( 4589): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-01 20:52:07.175+0900 D/AUL_PAD ( 4589): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-01 20:52:07.195+0900 D/AUL_PAD ( 4589): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-01 20:52:07.195+0900 D/AUL_PAD ( 4589): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-01 20:52:07.195+0900 D/AUL_PAD ( 4589): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-01 20:52:07.195+0900 D/AUL_PAD ( 4589): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-01 20:52:07.195+0900 D/LAUNCH  ( 4589): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-01 20:52:07.225+0900 I/CAPI_APPFW_APPLICATION( 4589): app_main.c: ui_app_main(697) > app_efl_main
04-01 20:52:07.225+0900 D/LAUNCH  ( 4589): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-01 20:52:07.265+0900 D/APP_CORE( 4589): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-01 20:52:07.275+0900 D/AUL     ( 4589): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 4589 is org.tizen.other
04-01 20:52:07.275+0900 D/APP_CORE( 4589): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-01 20:52:07.275+0900 D/APP_CORE( 4589): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-01 20:52:07.275+0900 D/AUL_PAD ( 2220): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-01 20:52:07.275+0900 D/AUL_PAD ( 2220): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-01 20:52:07.275+0900 D/AUL_PAD ( 2220): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-01 20:52:07.275+0900 D/RESOURCED( 2367): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-01 20:52:07.275+0900 D/RESOURCED( 2367): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2180
04-01 20:52:07.275+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 4589, type 4 
04-01 20:52:07.275+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 4589
04-01 20:52:07.275+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-01 20:52:07.275+0900 D/RESOURCED( 2367): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 4589
04-01 20:52:07.275+0900 D/RESOURCED( 2367): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 4589
04-01 20:52:07.275+0900 D/AUL     ( 4589): app_sock.c: __create_server_sock(135) > pg path - already exists
04-01 20:52:07.280+0900 D/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-01 20:52:07.280+0900 E/AUL     ( 2180): simple_util.c: __trm_app_info_send_socket(264) > access
04-01 20:52:07.280+0900 D/LAUNCH  ( 4589): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-01 20:52:07.280+0900 I/CAPI_APPFW_APPLICATION( 4589): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-01 20:52:07.280+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 4589
04-01 20:52:07.280+0900 D/RESOURCED( 2367): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-01 20:52:07.280+0900 D/RESOURCED( 2367): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-01 20:52:07.280+0900 D/AUL_AMD ( 2180): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-01 20:52:07.280+0900 D/RESOURCED( 2367): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-01 20:52:07.280+0900 D/RESOURCED( 2367): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-01 20:52:07.280+0900 D/RESOURCED( 2367): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-01 20:52:07.280+0900 D/RESOURCED( 2367): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-01 20:52:07.280+0900 E/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 592
04-01 20:52:07.280+0900 D/RESOURCED( 2367): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-01 20:52:07.470+0900 W/PROCESSMGR( 2099): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=4589
04-01 20:52:07.530+0900 D/indicator( 2222): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-01 20:52:07.530+0900 D/indicator( 2222): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2800003"
04-01 20:52:07.530+0900 D/indicator( 2222): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-01 20:52:07.530+0900 D/indicator( 2222): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2800003"
04-01 20:52:07.530+0900 D/indicator( 2222): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-01 20:52:07.550+0900 F/socket.io( 4589): thread_start
04-01 20:52:07.550+0900 F/socket.io( 4589): finish 0
04-01 20:52:07.550+0900 D/LAUNCH  ( 4589): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-01 20:52:07.560+0900 D/APP_CORE( 4589): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-01 20:52:07.560+0900 D/APP_CORE( 4589): appcore.c: __aul_handler(423) > [APP 4589]     AUL event: AUL_START
04-01 20:52:07.565+0900 D/APP_CORE( 4589): appcore-efl.c: __do_app(470) > [APP 4589] Event: RESET State: CREATED
04-01 20:52:07.565+0900 D/APP_CORE( 4589): appcore-efl.c: __do_app(496) > [APP 4589] RESET
04-01 20:52:07.565+0900 D/LAUNCH  ( 4589): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-01 20:52:07.565+0900 I/CAPI_APPFW_APPLICATION( 4589): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-01 20:52:07.565+0900 D/APP_SVC ( 4589): appsvc.c: __set_bundle(161) > __set_bundle
04-01 20:52:07.565+0900 D/LAUNCH  ( 4589): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-01 20:52:07.570+0900 I/APP_CORE( 4589): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-01 20:52:07.570+0900 I/APP_CORE( 4589): appcore-efl.c: __do_app(509) > [APP 4589] Initial Launching, call the resume_cb
04-01 20:52:07.570+0900 I/CAPI_APPFW_APPLICATION( 4589): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-01 20:52:07.570+0900 D/APP_CORE( 4589): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-01 20:52:07.575+0900 D/APP_CORE( 4589): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2800003
04-01 20:52:07.575+0900 D/APP_CORE( 4589): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2800003
04-01 20:52:07.720+0900 E/socket.io( 4589): 566: Connected.
04-01 20:52:07.725+0900 D/sio_packet( 4589): from json
04-01 20:52:07.725+0900 D/sio_packet( 4589): from json
04-01 20:52:07.725+0900 D/sio_packet( 4589): from json
04-01 20:52:07.725+0900 D/sio_packet( 4589): from json
04-01 20:52:07.725+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:07.725+0900 D/sio_packet( 4589): from json
04-01 20:52:07.725+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:07.725+0900 E/socket.io( 4589): 554: On handshake, sid
04-01 20:52:07.725+0900 E/socket.io( 4589): 651: Received Message type(connect)
04-01 20:52:07.725+0900 E/socket.io( 4589): 489: On Connected
04-01 20:52:07.725+0900 F/sio_packet( 4589): accept()
04-01 20:52:07.725+0900 E/socket.io( 4589): 743: encoded paylod length: 13
04-01 20:52:07.730+0900 E/socket.io( 4589): 800: ping exit, con is expired? 0, ec: Operation canceled
04-01 20:52:07.735+0900 D/APP_CORE( 2239): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-01 20:52:07.735+0900 D/APP_CORE( 2239): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-01 20:52:07.735+0900 D/APP_CORE( 2239): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-01 20:52:07.735+0900 D/APP_CORE( 2239): appcore-efl.c: __do_app(470) > [APP 2239] Event: PAUSE State: RUNNING
04-01 20:52:07.735+0900 D/APP_CORE( 2239): appcore-efl.c: __do_app(538) > [APP 2239] PAUSE
04-01 20:52:07.735+0900 I/CAPI_APPFW_APPLICATION( 2239): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-01 20:52:07.735+0900 D/MENU_SCREEN( 2239): menu_screen.c: _pause_cb(538) > Pause start
04-01 20:52:07.735+0900 D/APP_CORE( 2239): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-01 20:52:07.735+0900 E/APP_CORE( 2239): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-01 20:52:07.740+0900 D/APP_CORE( 4589): appcore.c: __prt_ltime(183) > [APP 4589] first idle after reset: 645 msec
04-01 20:52:07.740+0900 D/APP_CORE( 4589): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2800003 fully_obscured 0
04-01 20:52:07.740+0900 D/APP_CORE( 4589): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-01 20:52:07.740+0900 D/APP_CORE( 4589): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-01 20:52:07.740+0900 D/APP_CORE( 4589): appcore-efl.c: __do_app(470) > [APP 4589] Event: RESUME State: RUNNING
04-01 20:52:07.740+0900 D/LAUNCH  ( 4589): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-01 20:52:07.740+0900 D/LAUNCH  ( 4589): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-01 20:52:07.740+0900 D/LAUNCH  ( 4589): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-01 20:52:07.740+0900 D/APP_CORE( 4589): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-01 20:52:07.740+0900 E/APP_CORE( 4589): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-01 20:52:07.745+0900 D/RESOURCED( 2367): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2239, type = 2
04-01 20:52:07.745+0900 D/RESOURCED( 2367): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 4589, type = 0
04-01 20:52:07.745+0900 D/AUL_AMD ( 2180): amd_launch.c: __e17_status_handler(1888) > pid(4589) status(3)
04-01 20:52:07.745+0900 D/RESOURCED( 2367): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 4589
04-01 20:52:07.745+0900 I/RESOURCED( 2367): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 4589, oom : 200
04-01 20:52:07.745+0900 E/RESOURCED( 2367): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-01 20:52:07.810+0900 D/sio_packet( 4589): from json
04-01 20:52:07.810+0900 D/sio_packet( 4589): from json
04-01 20:52:07.810+0900 D/sio_packet( 4589): from json
04-01 20:52:07.810+0900 D/sio_packet( 4589): from json
04-01 20:52:07.810+0900 D/sio_packet( 4589): from json
04-01 20:52:07.810+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:07.810+0900 D/sio_packet( 4589): from json
04-01 20:52:07.810+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.810+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:07.810+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-01 20:52:07.815+0900 D/sio_packet( 4589): from json
04-01 20:52:07.815+0900 D/sio_packet( 4589): from json
04-01 20:52:07.815+0900 D/sio_packet( 4589): from json
04-01 20:52:07.815+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.880+0900 D/sio_packet( 4589): from json
04-01 20:52:07.880+0900 D/sio_packet( 4589): from json
04-01 20:52:07.880+0900 D/sio_packet( 4589): from json
04-01 20:52:07.880+0900 D/sio_packet( 4589): from json
04-01 20:52:07.880+0900 D/sio_packet( 4589): from json
04-01 20:52:07.880+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:07.880+0900 D/sio_packet( 4589): from json
04-01 20:52:07.880+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.880+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:07.890+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:07.905+0900 D/sio_packet( 4589): from json
04-01 20:52:07.905+0900 D/sio_packet( 4589): from json
04-01 20:52:07.905+0900 D/sio_packet( 4589): from json
04-01 20:52:07.905+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.915+0900 D/indicator( 2222): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-01 20:52:07.915+0900 D/indicator( 2222): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
04-01 20:52:07.930+0900 D/sio_packet( 4589): from json
04-01 20:52:07.930+0900 D/sio_packet( 4589): from json
04-01 20:52:07.930+0900 D/sio_packet( 4589): from json
04-01 20:52:07.930+0900 D/sio_packet( 4589): from json
04-01 20:52:07.930+0900 D/sio_packet( 4589): from json
04-01 20:52:07.930+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:07.930+0900 D/sio_packet( 4589): from json
04-01 20:52:07.930+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.930+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:07.945+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:07.945+0900 D/sio_packet( 4589): from json
04-01 20:52:07.945+0900 D/sio_packet( 4589): from json
04-01 20:52:07.945+0900 D/sio_packet( 4589): from json
04-01 20:52:07.945+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.985+0900 D/sio_packet( 4589): from json
04-01 20:52:07.985+0900 D/sio_packet( 4589): from json
04-01 20:52:07.985+0900 D/sio_packet( 4589): from json
04-01 20:52:07.985+0900 D/sio_packet( 4589): from json
04-01 20:52:07.985+0900 D/sio_packet( 4589): from json
04-01 20:52:07.985+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:07.985+0900 D/sio_packet( 4589): from json
04-01 20:52:07.985+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:07.985+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:07.995+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.000+0900 D/sio_packet( 4589): from json
04-01 20:52:08.000+0900 D/sio_packet( 4589): from json
04-01 20:52:08.000+0900 D/sio_packet( 4589): from json
04-01 20:52:08.000+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.050+0900 D/sio_packet( 4589): from json
04-01 20:52:08.050+0900 D/sio_packet( 4589): from json
04-01 20:52:08.050+0900 D/sio_packet( 4589): from json
04-01 20:52:08.050+0900 D/sio_packet( 4589): from json
04-01 20:52:08.050+0900 D/sio_packet( 4589): from json
04-01 20:52:08.050+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.050+0900 D/sio_packet( 4589): from json
04-01 20:52:08.050+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.050+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.060+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.060+0900 D/sio_packet( 4589): from json
04-01 20:52:08.060+0900 D/sio_packet( 4589): from json
04-01 20:52:08.060+0900 D/sio_packet( 4589): from json
04-01 20:52:08.060+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.115+0900 D/sio_packet( 4589): from json
04-01 20:52:08.115+0900 D/sio_packet( 4589): from json
04-01 20:52:08.115+0900 D/sio_packet( 4589): from json
04-01 20:52:08.115+0900 D/sio_packet( 4589): from json
04-01 20:52:08.115+0900 D/sio_packet( 4589): from json
04-01 20:52:08.115+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.115+0900 D/sio_packet( 4589): from json
04-01 20:52:08.115+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.115+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.125+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.130+0900 D/sio_packet( 4589): from json
04-01 20:52:08.130+0900 D/sio_packet( 4589): from json
04-01 20:52:08.130+0900 D/sio_packet( 4589): from json
04-01 20:52:08.130+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.165+0900 D/sio_packet( 4589): from json
04-01 20:52:08.165+0900 D/sio_packet( 4589): from json
04-01 20:52:08.165+0900 D/sio_packet( 4589): from json
04-01 20:52:08.165+0900 D/sio_packet( 4589): from json
04-01 20:52:08.165+0900 D/sio_packet( 4589): from json
04-01 20:52:08.165+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.165+0900 D/sio_packet( 4589): from json
04-01 20:52:08.165+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.170+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.175+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.175+0900 D/sio_packet( 4589): from json
04-01 20:52:08.175+0900 D/sio_packet( 4589): from json
04-01 20:52:08.175+0900 D/sio_packet( 4589): from json
04-01 20:52:08.175+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.225+0900 D/sio_packet( 4589): from json
04-01 20:52:08.225+0900 D/sio_packet( 4589): from json
04-01 20:52:08.225+0900 D/sio_packet( 4589): from json
04-01 20:52:08.225+0900 D/sio_packet( 4589): from json
04-01 20:52:08.225+0900 D/sio_packet( 4589): from json
04-01 20:52:08.225+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.225+0900 D/sio_packet( 4589): from json
04-01 20:52:08.225+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.225+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.235+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.240+0900 D/sio_packet( 4589): from json
04-01 20:52:08.240+0900 D/sio_packet( 4589): from json
04-01 20:52:08.240+0900 D/sio_packet( 4589): from json
04-01 20:52:08.240+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.280+0900 D/sio_packet( 4589): from json
04-01 20:52:08.280+0900 D/sio_packet( 4589): from json
04-01 20:52:08.280+0900 D/sio_packet( 4589): from json
04-01 20:52:08.280+0900 D/sio_packet( 4589): from json
04-01 20:52:08.280+0900 D/sio_packet( 4589): from json
04-01 20:52:08.280+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.280+0900 D/sio_packet( 4589): from json
04-01 20:52:08.280+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.280+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.285+0900 D/AUL_AMD ( 2180): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.other /opt/usr/apps/org.tizen.other/bin/other
04-01 20:52:08.285+0900 D/RUA     ( 2180): rua.c: rua_add_history(179) > rua_add_history start
04-01 20:52:08.290+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.290+0900 D/sio_packet( 4589): from json
04-01 20:52:08.290+0900 D/sio_packet( 4589): from json
04-01 20:52:08.290+0900 D/sio_packet( 4589): from json
04-01 20:52:08.290+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.300+0900 D/RUA     ( 2180): rua.c: rua_add_history(247) > rua_add_history ok
04-01 20:52:08.340+0900 D/sio_packet( 4589): from json
04-01 20:52:08.340+0900 D/sio_packet( 4589): from json
04-01 20:52:08.340+0900 D/sio_packet( 4589): from json
04-01 20:52:08.340+0900 D/sio_packet( 4589): from json
04-01 20:52:08.340+0900 D/sio_packet( 4589): from json
04-01 20:52:08.340+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.340+0900 D/sio_packet( 4589): from json
04-01 20:52:08.340+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.340+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.350+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.360+0900 D/sio_packet( 4589): from json
04-01 20:52:08.360+0900 D/sio_packet( 4589): from json
04-01 20:52:08.360+0900 D/sio_packet( 4589): from json
04-01 20:52:08.360+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.395+0900 D/sio_packet( 4589): from json
04-01 20:52:08.395+0900 D/sio_packet( 4589): from json
04-01 20:52:08.395+0900 D/sio_packet( 4589): from json
04-01 20:52:08.395+0900 D/sio_packet( 4589): from json
04-01 20:52:08.395+0900 D/sio_packet( 4589): from json
04-01 20:52:08.395+0900 D/sio_packet( 4589): IsInt64
04-01 20:52:08.395+0900 D/sio_packet( 4589): from json
04-01 20:52:08.395+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.395+0900 F/get_binary( 4589): in get binary_message()...
04-01 20:52:08.405+0900 D/CAPI_MEDIA_IMAGE_UTIL( 4589): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-01 20:52:08.405+0900 D/sio_packet( 4589): from json
04-01 20:52:08.405+0900 D/sio_packet( 4589): from json
04-01 20:52:08.405+0900 D/sio_packet( 4589): from json
04-01 20:52:08.405+0900 E/socket.io( 4589): 669: Received Message type(Event)
04-01 20:52:08.625+0900 W/CRASH_MANAGER( 4600): worker.c: worker_job(1189) > 11045896f7468142788912
