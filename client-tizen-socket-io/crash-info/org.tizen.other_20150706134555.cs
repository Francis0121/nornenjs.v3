S/W Version Information
Model: SM-Z130H
Tizen-Version: 2.3.0.1
Build-Number: Z130HDDU0BOD7
Build-Date: 2015.04.16 12:41:35

Crash Information
Process Name: other
PID: 12380
Date: 2015-07-06 13:45:55+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xb12ff004

Register Information
r0   = 0xb12ff008, r1   = 0xb4d40bf3
r2   = 0x000000e4, r3   = 0x00000000
r4   = 0xb4de7f1c, r5   = 0xb12ff008
r6   = 0x00000245, r7   = 0xbef82d88
r8   = 0xbef82e14, r9   = 0xb6f9f2c4
r10  = 0x00000000, fp   = 0xb7c6c5a0
ip   = 0xb4de7f98, sp   = 0xbef82d70
lr   = 0xb4d40bf3, pc   = 0xb6cd6150
cpsr = 0xa0000010

Memory Information
MemTotal:   730748 KB
MemFree:    213144 KB
Buffers:     28944 KB
Cached:     211696 KB
VmPeak:      99412 KB
VmSize:      99408 KB
VmLck:           0 KB
VmPin:           0 KB
VmHWM:       22240 KB
VmRSS:       22236 KB
VmData:      45028 KB
VmStk:         136 KB
VmExe:          20 KB
VmLib:       25912 KB
VmPTE:          68 KB
VmSwap:          0 KB

Threads Information
Threads: 5
PID = 12380 TID = 12380
12380 12386 12387 12389 12390 

Maps Information
b2001000 b2800000 rw-p [stack:12390]
b2986000 b3185000 rw-p [stack:12389]
b3186000 b3985000 rw-p [stack:12388]
b3a18000 b3a19000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b3a21000 b3a28000 r-xp /usr/lib/libfeedback.so.0.1.4
b3a48000 b3a49000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b3a51000 b3a52000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b3a5a000 b3a71000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b3c18000 b3c1c000 r-xp /usr/lib/bufmgr/libtbm_sprd7727.so.0.0.0
b3c26000 b4425000 rw-p [stack:12386]
b4425000 b456c000 r-xp /usr/lib/driver/libMali.so
b4579000 b45c9000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b4900000 b49ca000 r-xp /usr/lib/libCOREGL.so.4.0
b49db000 b4b1a000 r-xp /usr/lib/libicui18n.so.51.1
b4b2a000 b4b2f000 r-xp /usr/lib/libxcb-render.so.0.0.0
b4b38000 b4b39000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b4b42000 b4b5a000 r-xp /usr/lib/libpng12.so.0.50.0
b4b62000 b4b65000 r-xp /usr/lib/libEGL.so.1.4
b4b6d000 b4b7b000 r-xp /usr/lib/libGLESv2.so.2.0
b4b84000 b4b86000 r-xp /usr/lib/libiniparser.so.0
b4b90000 b4b98000 r-xp /usr/lib/libui-extension.so.0.1.0
b4b99000 b4b9c000 r-xp /usr/lib/libnative-buffer.so.0.1.0
b4ba4000 b4c5a000 r-xp /usr/lib/libcairo.so.2.11200.14
b4c65000 b4c77000 r-xp /usr/lib/libtts.so
b4c7f000 b4c86000 r-xp /usr/lib/libtbm.so.1.0.0
b4c8e000 b4c93000 r-xp /usr/lib/libcapi-media-tool.so.0.1.3
b4c9b000 b4c9f000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b4ca8000 b4caa000 r-xp /usr/lib/libdri2.so.0.0.0
b4cb2000 b4cb9000 r-xp /usr/lib/libdrm.so.2.4.0
b4cc2000 b4cd8000 r-xp /usr/lib/libefl-assist.so.0.1.0
b4ce0000 b4ce8000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b4cf0000 b4cf5000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b4cfd000 b4cff000 r-xp /usr/lib/libefl-extension.so.0.1.0
b4d07000 b4d0e000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.4
b4d17000 b4d1a000 r-xp /usr/lib/libcapi-media-image-util.so.0.3.22
b4d24000 b4d2e000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b4d37000 b4ddf000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b4de9000 b4df3000 r-xp /lib/libnss_files-2.13.so
b4dfc000 b4e0e000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b4e16000 b4e2c000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b4e34000 b4f02000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4f19000 b4f3d000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4f46000 b4f4c000 r-xp /usr/lib/libappsvc.so.0.1.0
b4f54000 b4f62000 r-xp /usr/lib/libail.so.0.1.0
b4f6a000 b4f6c000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.3
b4f75000 b4f7a000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.3
b4f82000 b4f84000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4f8d000 b4f8e000 r-xp /usr/lib/libosp-env-config.so.1.2.2.1
b4f97000 b4f9b000 r-xp /usr/lib/libogg.so.0.7.1
b4fa3000 b4fc5000 r-xp /usr/lib/libvorbis.so.0.4.3
b4fcd000 b50b1000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b50c5000 b50f6000 r-xp /usr/lib/libFLAC.so.8.2.0
b5a90000 b5b24000 r-xp /usr/lib/libstdc++.so.6.0.16
b5b37000 b5b39000 r-xp /usr/lib/libXau.so.6.0.0
b5b41000 b5b4b000 r-xp /usr/lib/libspdy.so.0.0.0
b5b54000 b5ba0000 r-xp /usr/lib/libssl.so.1.0.0
b5bad000 b5bdb000 r-xp /usr/lib/libidn.so.11.5.44
b5be3000 b5bed000 r-xp /usr/lib/libcares.so.2.1.0
b5bf5000 b5c16000 r-xp /usr/lib/libexif.so.12.3.3
b5c29000 b5c6e000 r-xp /usr/lib/libsndfile.so.1.0.25
b5c7c000 b5c92000 r-xp /lib/libexpat.so.1.5.2
b5c9b000 b5d80000 r-xp /usr/lib/libicuuc.so.51.1
b5d95000 b5dc9000 r-xp /usr/lib/libicule.so.51.1
b5dd2000 b5de5000 r-xp /usr/lib/libxcb.so.1.1.0
b5ded000 b5e2a000 r-xp /usr/lib/libcurl.so.4.3.0
b5e33000 b5e3c000 r-xp /usr/lib/libethumb.so.1.7.99
b5e45000 b5e47000 r-xp /usr/lib/libctxdata.so.0.0.0
b5e4f000 b5e5c000 r-xp /usr/lib/libremix.so.0.0.0
b5e64000 b5e65000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5e6d000 b5e84000 r-xp /usr/lib/liblua-5.1.so
b5e8d000 b5e94000 r-xp /usr/lib/libembryo.so.1.7.99
b5e9c000 b5ebf000 r-xp /usr/lib/libjpeg.so.8.0.2
b5ed7000 b5f2d000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5f3a000 b5f8d000 r-xp /usr/lib/libfreetype.so.6.8.1
b5f98000 b5fc0000 r-xp /usr/lib/libfontconfig.so.1.8.0
b5fc1000 b6007000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b6010000 b6023000 r-xp /usr/lib/libfribidi.so.0.3.1
b602b000 b602e000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b6036000 b603a000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b6042000 b6047000 r-xp /usr/lib/libecore_fb.so.1.7.99
b6050000 b605a000 r-xp /usr/lib/libXext.so.6.4.0
b6062000 b6143000 r-xp /usr/lib/libX11.so.6.3.0
b614e000 b6151000 r-xp /usr/lib/libXtst.so.6.1.0
b6159000 b615f000 r-xp /usr/lib/libXrender.so.1.3.0
b6167000 b616c000 r-xp /usr/lib/libXrandr.so.2.2.0
b6174000 b6175000 r-xp /usr/lib/libXinerama.so.1.0.0
b617e000 b6186000 r-xp /usr/lib/libXi.so.6.1.0
b6187000 b618a000 r-xp /usr/lib/libXfixes.so.3.1.0
b6192000 b6194000 r-xp /usr/lib/libXgesture.so.7.0.0
b619c000 b619e000 r-xp /usr/lib/libXcomposite.so.1.0.0
b61a6000 b61a7000 r-xp /usr/lib/libXdamage.so.1.1.0
b61b0000 b61b6000 r-xp /usr/lib/libXcursor.so.1.0.2
b61bf000 b61d8000 r-xp /usr/lib/libecore_con.so.1.7.99
b61e2000 b61e8000 r-xp /usr/lib/libecore_imf.so.1.7.99
b61f0000 b61f8000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6200000 b6209000 r-xp /usr/lib/libedbus.so.1.7.99
b6211000 b626e000 r-xp /usr/lib/libedje.so.1.7.99
b6277000 b6288000 r-xp /usr/lib/libecore_input.so.1.7.99
b6290000 b6295000 r-xp /usr/lib/libecore_file.so.1.7.99
b629d000 b62b6000 r-xp /usr/lib/libeet.so.1.7.99
b62c7000 b62cb000 r-xp /usr/lib/libappcore-common.so.1.1
b62d3000 b639a000 r-xp /usr/lib/libevas.so.1.7.99
b63bf000 b63e0000 r-xp /usr/lib/libecore_evas.so.1.7.99
b63e9000 b6418000 r-xp /usr/lib/libecore_x.so.1.7.99
b6422000 b6559000 r-xp /usr/lib/libelementary.so.1.7.99
b656f000 b6570000 r-xp /usr/lib/osp/libappinfo.so.1.2.2.1
b6578000 b657c000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.3
b6587000 b658a000 r-xp /lib/libgpg-error.so.0.5.0
b6592000 b65ea000 r-xp /usr/lib/libgcrypt.so.11.5.3
b65f4000 b6620000 r-xp /usr/lib/libsystemd.so.0.0.1
b6629000 b662b000 r-xp /usr/lib/journal/libjournal.so.0.1.0
b6634000 b66ff000 r-xp /usr/lib/libxml2.so.2.7.8
b670d000 b671d000 r-xp /lib/libresolv-2.13.so
b6721000 b6737000 r-xp /lib/libz.so.1.2.5
b673f000 b6741000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b6749000 b674e000 r-xp /usr/lib/libffi.so.5.0.10
b6757000 b6758000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6760000 b6763000 r-xp /lib/libattr.so.1.1.0
b676b000 b676e000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6776000 b677d000 r-xp /usr/lib/libvconf.so.0.2.45
b6786000 b692e000 r-xp /usr/lib/libcrypto.so.1.0.0
b694f000 b6965000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b696d000 b69d6000 r-xp /lib/libm-2.13.so
b69df000 b6a1f000 r-xp /usr/lib/libeina.so.1.7.99
b6a28000 b6a5c000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a65000 b6b39000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b45000 b6b4a000 r-xp /usr/lib/libcapi-base-common.so.0.1.6
b6b53000 b6b59000 r-xp /lib/librt-2.13.so
b6b62000 b6b69000 r-xp /lib/libcrypt-2.13.so
b6b99000 b6b9c000 r-xp /lib/libcap.so.2.21
b6ba4000 b6ba6000 r-xp /usr/lib/libiri.so
b6bae000 b6bcd000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6bd5000 b6beb000 r-xp /usr/lib/libecore.so.1.7.99
b6c01000 b6c06000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6c0f000 b6c26000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c2f000 b6c3a000 r-xp /lib/libunwind.so.8.0.1
b6c67000 b6d82000 r-xp /lib/libc-2.13.so
b6d90000 b6d98000 r-xp /lib/libgcc_s-4.6.4.so.1
b6da0000 b6dca000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6dd3000 b6dd6000 r-xp /usr/lib/libbundle.so.0.1.22
b6dde000 b6de0000 r-xp /lib/libdl-2.13.so
b6de9000 b6dec000 r-xp /usr/lib/libsmack.so.1.0.0
b6df4000 b6ec4000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6ec5000 b6f2a000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f34000 b6f46000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f4e000 b6f62000 r-xp /lib/libpthread-2.13.so
b6f6d000 b6f6f000 r-xp /usr/lib/libdlog.so.0.0.0
b6f77000 b6f82000 r-xp /usr/lib/libaul.so.0.1.0
b6f91000 b6f94000 rw-p [stack:12387]
b6f94000 b6f97000 r-xp /usr/lib/libappcore-efl.so.1.1
b6fa1000 b6fa5000 r-xp /usr/lib/libsys-assert.so
b6fae000 b6fcb000 r-xp /lib/ld-2.13.so
b6fd4000 b6fd9000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b7c5a000 b7c84000 rw-p [heap]
b7c84000 b81f0000 rw-p [heap]
bef63000 bef84000 rw-p [stack]
bef63000 bef84000 rw-p [stack]
End of Maps Information

Callstack Information (PID:12380)
Call Stack Count: 10
 0: cfree + 0x30 (0xb6cd6150) [/lib/libc.so.6] + 0x6f150
 1: app_terminate + 0x2a (0xb4d40bf3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9bf3
 2: (0xb6579ad1) [/usr/lib/libcapi-appfw-application.so.0] + 0x1ad1
 3: appcore_efl_main + 0x1c4 (0xb6f963cd) [/usr/lib/libappcore-efl.so.1] + 0x23cd
 4: ui_app_main + 0xb0 (0xb657a499) [/usr/lib/libcapi-appfw-application.so.0] + 0x2499
 5: main + 0x10c (0xb4d40d95) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9d95
 6: (0xb6fd6dc7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2dc7
 7: (0xb6fd5d8f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1d8f
 8: __libc_start_main + 0x114 (0xb6c7e82c) [/lib/libc.so.6] + 0x1782c
 9: (0xb6fd60d4) [/opt/usr/apps/org.tizen.other/bin/other] + 0x20d4
End of Call Stack

Package Information
Package Name: org.tizen.other
Package ID : org.tizen.other
Version: 1.0.0
Package Type: rpm
App Name: Nornenjs
App ID: org.tizen.other
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
34): (1584) > Package = [org.tizen.nornenjs], Key = [install_percent], Value = [60], install = [1]
07-06 13:44:16.360+0900 E/PKGMGR_CERT(12040): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 2 92
07-06 13:44:16.360+0900 E/PKGMGR_CERT(12040): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 21 4
07-06 13:44:16.360+0900 E/PKGMGR_CERT(12040): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 22 4
07-06 13:44:16.360+0900 E/PKGMGR_CERT(12040): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 23 4
07-06 13:44:16.360+0900 E/PKGMGR_CERT(12040): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 24 4
07-06 13:44:16.610+0900 E/PKGMGR_CERT(12040): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(571) > Transaction Commit and End
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_make_directory(1927) > mkdir failed. appdir=[/usr/apps/org.tizen.nornenjs/shared], errno=[2][No such file or directory]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1549) > skip! empty dirpath=[/usr/apps/org.tizen.nornenjs/shared]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1549) > skip! empty dirpath=[/opt/usr/apps/org.tizen.nornenjs/shared/data]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1549) > skip! empty dirpath=[/usr/apps/org.tizen.nornenjs/shared/res]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_file_policy(1534) > skip! empty filepath=[/usr/apps/org.tizen.nornenjs/tizen-manifest.xml]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_file_policy(1534) > skip! empty filepath=[/usr/apps/org.tizen.nornenjs/author-signature.xml]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_file_policy(1534) > skip! empty filepath=[/usr/apps/org.tizen.nornenjs/signature1.xml]
07-06 13:44:16.620+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_apply_file_policy(1534) > skip! empty filepath=[/usr/share/packages/org.tizen.nornenjs.xml]
07-06 13:44:17.090+0900 E/rpm-installer(12040): coretpk-installer.c: _coretpk_installer_get_smack_label_access(1099) > Error in getting smack ACCESS label failed. result[-1] (path:[/opt/usr/apps/org.tizen.nornenjs/shared/data]))
07-06 13:44:17.830+0900 E/PKGMGR_SERVER(12039): pkgmgr-server.c: exit_server(887) > exit_server Start [backend_status=0, queue_status=1, drm_status=1] 
07-06 13:44:18.720+0900 I/Tizen::App( 1034): (1894) > PackageEventHandler - req: 1, pkg_type: rpm, pkg_name: org.tizen.nornenjs, key: install_percent, val: 100
07-06 13:44:18.720+0900 I/Tizen::App( 1034): (119) > InstallationInProgress [100]
07-06 13:44:18.720+0900 I/Tizen::App( 1034): (1584) > Package = [org.tizen.nornenjs], Key = [install_percent], Value = [100], install = [1]
07-06 13:44:18.730+0900 I/Tizen::App( 1034): (1894) > PackageEventHandler - req: 1, pkg_type: rpm, pkg_name: org.tizen.nornenjs, key: end, val: ok
07-06 13:44:18.730+0900 I/Tizen::App( 1034): (78) > Installation is Completed. [Package = org.tizen.nornenjs]
07-06 13:44:18.730+0900 I/Tizen::App( 1034): (663) > Enter. package(org.tizen.nornenjs), installationResult(0)
07-06 13:44:18.740+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: OnClientListenCb(539) >  #Step 1
07-06 13:44:18.740+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: OnClientListenCb(543) >  #Step 2
07-06 13:44:18.740+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _GetAppIds(344) >  BEGIN
07-06 13:44:18.780+0900 I/Tizen::App( 1034): (1360) > package(org.tizen.nornenjs), version(1.0.0), type(rpm), displayName(Nornenjs), uninstallable(1), downloaded(1), updated(1), preloaded(0)movable(1), externalStorage(0), mainApp(org.tizen.nornenjs), storeClient(), appRootPath(/opt/usr/apps/org.tizen.nornenjs)
07-06 13:44:18.780+0900 E/PKGMGR_INFO(12040): pkgmgrinfo_feature.c: pkgmgrinfo_updateinfo_remove(528) > (strstr(vconf_str, pkgid_is) == NULL) pkgid is already removed
07-06 13:44:18.780+0900 E/PKGMGR_INSTALLER(12040): pkgmgr_installer.c: __send_event(114) > fail to remove update-service
07-06 13:44:18.860+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _PkgMgrAppInfoGetListCb(246) >  ##### [org.tizen.nornenjs]
07-06 13:44:18.860+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _GetAppIds(369) >  ##### [org.tizen.nornenjs]
07-06 13:44:18.860+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _GetAppIds(373) >  END
07-06 13:44:18.860+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _DoPkgJob(452) >  #Step 3 size[1]
07-06 13:44:18.860+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _DoPkgJob(456) >  appId[org.tizen.nornenjs]
07-06 13:44:18.860+0900 I/Tizen::App( 1034): (483) > pkgmgrinfo_appinfo_get_appid(): app = [org.tizen.nornenjs]
07-06 13:44:18.880+0900 I/Tizen::App( 1034): (416) > appName = [nornenjs]
07-06 13:44:18.880+0900 I/Tizen::App( 1034): (509) > exe = [/opt/usr/apps/org.tizen.nornenjs/bin/nornenjs], displayName = [Nornenjs], mainApp = [1], menuIconVisible = [0], serviceApp = [0]
07-06 13:44:18.880+0900 E/PKGMGR_INFO( 1034): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_list(717) > (component == PMINFO_SVC_APP) PMINFO_SVC_APP is done
07-06 13:44:18.880+0900 I/Tizen::App( 1034): (675) > Application count(1) in this package
07-06 13:44:18.880+0900 I/Tizen::App( 1034): (695) > Exit.
07-06 13:44:18.880+0900 I/Tizen::App( 1034): (1584) > Package = [org.tizen.nornenjs], Key = [end], Value = [ok], install = [1]
07-06 13:44:18.880+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: _GetAppInfo(945) >  AppId[org.tizen.nornenjs] Name[Nornenjs] Icon[/opt/usr/apps/org.tizen.nornenjs/shared/res/favicon.png] enable[1] system[0]
07-06 13:44:18.880+0900 I/Tizen::App( 1034): (855) > Enter.
07-06 13:44:18.880+0900 E/cluster-home(  606): mainmenu-package-manager.cpp: GetAppInfo(599) >  Find a App Info AppId[org.tizen.nornenjs] Name[Nornenjs] Icon[/opt/usr/apps/org.tizen.nornenjs/shared/res/favicon.png] enable[1] system[0]
07-06 13:44:18.890+0900 I/Tizen::App( 1034): (416) > appName = [nornenjs]
07-06 13:44:18.890+0900 I/Tizen::App( 1034): (509) > exe = [/opt/usr/apps/org.tizen.nornenjs/bin/nornenjs], displayName = [Nornenjs], mainApp = [1], menuIconVisible = [0], serviceApp = [0]
07-06 13:44:18.890+0900 I/Tizen::App( 1034): (2343) > info file is not existed. [/opt/usr/apps/org.tizen./info/org.tizen.nornenjs.info]
07-06 13:44:18.890+0900 I/Tizen::App( 1034): (131) > Enter
07-06 13:44:18.900+0900 I/Tizen::App( 1034): (137) > org.tizen.nornenjs does not have launch condition
07-06 13:44:18.900+0900 I/Tizen::App( 1034): (898) > Exit.
07-06 13:44:19.830+0900 E/PKGMGR_SERVER(12039): pkgmgr-server.c: exit_server(887) > exit_server Start [backend_status=1, queue_status=1, drm_status=1] 
07-06 13:44:19.830+0900 E/PKGMGR_SERVER(12039): pkgmgr-server.c: main(1704) > package manager server terminated.
07-06 13:44:20.190+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 0
07-06 13:44:20.200+0900 I/AUL     (  452): menu_db_util.h: _get_app_info_from_db_by_apppath(240) > path : /usr/bin/launch_app, ret : 0
07-06 13:44:20.200+0900 I/AUL     (  452): menu_db_util.h: _get_app_info_from_db_by_apppath(240) > path : /bin/bash, ret : 0
07-06 13:44:20.200+0900 E/AUL_AMD (  452): amd_appinfo.c: appinfo_get_value(791) > appinfo get value: Invalid argument, 24
07-06 13:44:20.280+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: ui_app_main(699) > app_efl_main
07-06 13:44:20.320+0900 I/UXT     (12113): uxt_object_manager.cpp: on_initialized(287) > Initialized.
07-06 13:44:20.330+0900 E/RESOURCED(  768): proc-main.c: resourced_proc_status_change(614) > [resourced_proc_status_change,614] available memory = 463
07-06 13:44:20.340+0900 I/Tizen::App( 1047): (733) > Finished invoking application event listener for org.tizen.nornenjs, 12113.
07-06 13:44:20.350+0900 I/Tizen::App( 1034): (499) > LaunchedApp(org.tizen.nornenjs)
07-06 13:44:20.350+0900 I/Tizen::App( 1034): (733) > Finished invoking application event listener for org.tizen.nornenjs, 12113.
07-06 13:44:20.380+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: _ui_app_appcore_create(560) > app_appcore_create
07-06 13:44:20.580+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: _ui_app_appcore_reset(642) > app_appcore_reset
07-06 13:44:20.590+0900 I/APP_CORE(12113): appcore-efl.c: __do_app(509) > Legacy lifecycle: 0
07-06 13:44:20.590+0900 I/APP_CORE(12113): appcore-efl.c: __do_app(511) > [APP 12113] Initial Launching, call the resume_cb
07-06 13:44:20.590+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: _ui_app_appcore_resume(624) > app_appcore_resume
07-06 13:44:20.630+0900 W/APP_CORE(12113): appcore-efl.c: __show_cb(822) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:4600002
07-06 13:44:20.630+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=12113
07-06 13:44:20.670+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 13:44:20.680+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = org.tizen.nornenjs, pid = 12113"
07-06 13:44:20.710+0900 I/MALI    (  606): egl_platform_x11_tizen.c: tizen_update_native_surface_private(181) > [EGL-X11] surface->[0xb06357f8] swap changed from sync to async
07-06 13:44:20.750+0900 I/CAPI_APPFW_APPLICATION(  606): app_main.c: app_appcore_pause(202) > app_appcore_pause
07-06 13:44:20.750+0900 E/cluster-home(  606): homescreen-main.cpp: app_pause(355) >  app pause
07-06 13:44:20.770+0900 I/Tizen::System( 1047): (259) > Active app [org.tizen.], current [com.samsun] 
07-06 13:44:20.770+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 13:44:20.780+0900 W/AUL_AMD (  452): amd_key.c: _key_ungrab(250) > fail(-1) to ungrab key(XF86Stop)
07-06 13:44:20.780+0900 W/AUL_AMD (  452): amd_launch.c: __e17_status_handler(2132) > back key ungrab error
07-06 13:44:20.790+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 13:44:23.580+0900 I/nornenjs(12113): Timer expired after 3.001 seconds.
07-06 13:44:23.580+0900 E/EFL     (12113): ecore<12113> ecore.c:568 _ecore_magic_fail() 
07-06 13:44:23.580+0900 E/EFL     (12113): *** ECORE ERROR: Ecore Magic Check Failed!!!
07-06 13:44:23.580+0900 E/EFL     (12113): *** IN FUNCTION: ecore_timer_delay()
07-06 13:44:23.580+0900 E/EFL     (12113): ecore<12113> ecore.c:570 _ecore_magic_fail()   Input handle pointer is NULL!
07-06 13:44:23.580+0900 E/EFL     (12113): ecore<12113> ecore.c:581 _ecore_magic_fail() *** NAUGHTY PROGRAMMER!!!
07-06 13:44:23.580+0900 E/EFL     (12113): *** SPANK SPANK SPANK!!!
07-06 13:44:23.580+0900 E/EFL     (12113): *** Now go fix your code. Tut tut tut!
07-06 13:44:23.610+0900 E/EFL     (12113): evas_main<12113> evas_font_dir.c:70 _evas_font_init_instance() ENTER:: evas_font_init
07-06 13:44:23.620+0900 E/EFL     (12113): evas_main<12113> evas_font_dir.c:90 evas_font_init() DONE:: evas_font_init
07-06 13:44:27.930+0900 D/nornenjs(12113): create_volume_render_view 01
07-06 13:44:27.930+0900 D/nornenjs(12113): create_volume_render_view 02
07-06 13:44:27.930+0900 E/EFL     (12113): evas_main<12113> evas_gl.c:153 evas_gl_new() Evas GL engine not available.
07-06 13:44:27.930+0900 E/EFL     (12113): elementary<12113> elm_glview.c:205 _elm_glview_smart_add() Failed Creating an Evas GL Object.
07-06 13:44:27.930+0900 D/nornenjs(12113): create_volume_render_view 03
07-06 13:44:27.930+0900 D/nornenjs(12113): create_volume_render_view 04
07-06 13:44:27.990+0900 E/socket.io(12113): 566: Connected.
07-06 13:44:27.990+0900 E/socket.io(12113): 554: On handshake, sid
07-06 13:44:27.990+0900 E/socket.io(12113): 651: Received Message type(connect)
07-06 13:44:27.990+0900 E/socket.io(12113): 489: On Connected
07-06 13:44:27.990+0900 F/sio_packet(12113): accept()
07-06 13:44:27.990+0900 E/socket.io(12113): 743: encoded paylod length: 18
07-06 13:44:28.010+0900 E/socket.io(12113): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:44:28.090+0900 E/socket.io(12113): 669: Received Message type(Event)
07-06 13:44:28.090+0900 F/sio_packet(12113): accept()
07-06 13:44:28.090+0900 E/socket.io(12113): 743: encoded paylod length: 21
07-06 13:44:28.090+0900 E/socket.io(12113): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:44:28.130+0900 E/socket.io(12113): 669: Received Message type(Event)
07-06 13:44:28.130+0900 F/get_binary(12113): in get binary_message()...
07-06 13:44:53.090+0900 F/sio_packet(12113): accept()
07-06 13:45:18.090+0900 F/sio_packet(12113): accept()
07-06 13:45:43.090+0900 F/sio_packet(12113): accept()
07-06 13:45:43.600+0900 W/STARTER (  525): hw_key.c: _key_press_cb(673) > [_key_press_cb:673] Home Key is pressed
07-06 13:45:43.600+0900 W/STARTER (  525): hw_key.c: _key_press_cb(691) > [_key_press_cb:691] homekey count : 1
07-06 13:45:43.710+0900 W/STARTER (  525): hw_key.c: _key_release_cb(503) > [_key_release_cb:503] Home Key is released
07-06 13:45:43.730+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
07-06 13:45:43.730+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
07-06 13:45:43.730+0900 E/VOLUME  (  595): volume_x_event.c: volume_x_input_event_unregister(351) > [volume_x_input_event_unregister:351] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
07-06 13:45:43.730+0900 E/VOLUME  (  595): volume_control.c: volume_control_close(708) > [volume_control_close:708] Failed to unregister x input event handler
07-06 13:45:43.740+0900 E/VOLUME  (  595): volume_view.c: volume_view_setting_icon_callback_del(533) > [volume_view_setting_icon_callback_del:533] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
07-06 13:45:43.950+0900 W/STARTER (  525): hw_key.c: _homekey_timer_cb(404) > [_homekey_timer_cb:404] _homekey_timer_cb, homekey count[1], lock state[0]
07-06 13:45:43.960+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 0
07-06 13:45:43.970+0900 I/AUL     (  452): menu_db_util.h: _get_app_info_from_db_by_apppath(240) > path : /usr/bin/starter, ret : 0
07-06 13:45:43.970+0900 E/AUL_AMD (  452): amd_appinfo.c: appinfo_get_value(791) > appinfo get value: Invalid argument, 24
07-06 13:45:43.980+0900 W/AUL_AMD (  452): amd_launch.c: __nofork_processing(1083) > __nofork_processing, cmd: 0, pid: 606
07-06 13:45:43.990+0900 I/CAPI_APPFW_APPLICATION(  606): app_main.c: app_appcore_reset(245) > app_appcore_reset
07-06 13:45:43.990+0900 I/APP_CORE(  606): appcore-efl.c: __do_app(516) > Legacy lifecycle: 1
07-06 13:45:43.990+0900 W/AUL_AMD (  452): amd_request.c: __send_home_launch_signal(441) > send a home launch signal
07-06 13:45:43.990+0900 W/AUL_AMD (  452): amd_launch.c: __reply_handler(851) > listen fd(32) , send fd(14), pid(606), cmd(0)
07-06 13:45:43.990+0900 W/AUL_AMD (  452): amd_key.c: _key_ungrab(250) > fail(-1) to ungrab key(XF86Stop)
07-06 13:45:43.990+0900 W/AUL_AMD (  452): amd_launch.c: __e17_status_handler(2132) > back key ungrab error
07-06 13:45:44.000+0900 E/STARTER (  525): dbus-util.c: starter_dbus_home_raise_signal_send(168) > [starter_dbus_home_raise_signal_send:168] Sending HOME RAISE signal, result:0
07-06 13:45:44.060+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=606
07-06 13:45:44.080+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 13:45:44.140+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: _ui_app_appcore_pause(607) > app_appcore_pause
07-06 13:45:44.160+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = com.samsung.homescreen, pid = 606"
07-06 13:45:44.180+0900 I/Tizen::System( 1047): (259) > Active app [com.samsun], current [org.tizen.] 
07-06 13:45:44.180+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 13:45:44.190+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 13:45:44.420+0900 I/RESOURCED(  768): logging.c: broadcast_logging_data_updated_signal(714) > [broadcast_logging_data_updated_signal,714] broadcast logging_data updated signal!
07-06 13:45:45.510+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 13:45:45.510+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 13:45:45.510+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 13:45:45.510+0900 W/cluster-home(  606): cluster-data-provider.cpp: OnFocusedViewChanged(6531) >  view type is not chaged,same view[0]
07-06 13:45:46.240+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 13:45:46.240+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 13:45:46.240+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 13:45:46.420+0900 I/MALI    (  606): egl_platform_x11_tizen.c: tizen_update_native_surface_private(172) > [EGL-X11] surface->[0xb06357f8] swap changed from async to sync
07-06 13:45:46.890+0900 W/cluster-view(  606): mainmenu-apps-view-impl.cpp: _OnScrollComplete(2041) >  booster timer is still running on apps-view, Stop boost timer!!!
07-06 13:45:47.330+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 1
07-06 13:45:47.460+0900 E/RESOURCED(  768): proc-main.c: find_pid_info(96) > [find_pid_info,96] Please provide valid pointer.
07-06 13:45:47.460+0900 I/Tizen::App( 1034): (499) > LaunchedApp(org.tizen.other)
07-06 13:45:47.460+0900 I/Tizen::App( 1034): (733) > Finished invoking application event listener for org.tizen.other, 12380.
07-06 13:45:47.460+0900 E/RESOURCED(  768): proc-main.c: resourced_proc_status_change(614) > [resourced_proc_status_change,614] available memory = 459
07-06 13:45:47.480+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: ui_app_main(699) > app_efl_main
07-06 13:45:47.500+0900 I/Tizen::App( 1047): (733) > Finished invoking application event listener for org.tizen.other, 12380.
07-06 13:45:47.550+0900 I/UXT     (12380): uxt_object_manager.cpp: on_initialized(287) > Initialized.
07-06 13:45:47.600+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_create(560) > app_appcore_create
07-06 13:45:47.880+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=12380
07-06 13:45:47.920+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 13:45:47.930+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = org.tizen.other, pid = 12380"
07-06 13:45:47.930+0900 I/MALI    (  606): egl_platform_x11_tizen.c: tizen_update_native_surface_private(181) > [EGL-X11] surface->[0xb06357f8] swap changed from sync to async
07-06 13:45:47.980+0900 I/Tizen::System( 1047): (259) > Active app [org.tizen.], current [com.samsun] 
07-06 13:45:47.980+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 13:45:47.990+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 13:45:48.040+0900 E/EFL     (12380): evas_main<12380> evas_font_dir.c:70 _evas_font_init_instance() ENTER:: evas_font_init
07-06 13:45:48.060+0900 E/EFL     (12380): evas_main<12380> evas_font_dir.c:90 evas_font_init() DONE:: evas_font_init
07-06 13:45:48.070+0900 F/socket.io(12380): thread_start
07-06 13:45:48.070+0900 F/socket.io(12380): finish 0
07-06 13:45:48.070+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_reset(642) > app_appcore_reset
07-06 13:45:48.080+0900 I/APP_CORE(12380): appcore-efl.c: __do_app(509) > Legacy lifecycle: 0
07-06 13:45:48.080+0900 I/APP_CORE(12380): appcore-efl.c: __do_app(511) > [APP 12380] Initial Launching, call the resume_cb
07-06 13:45:48.080+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_resume(624) > app_appcore_resume
07-06 13:45:48.080+0900 W/APP_CORE(12380): appcore-efl.c: __show_cb(822) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:4800003
07-06 13:45:48.140+0900 I/CAPI_APPFW_APPLICATION(  606): app_main.c: app_appcore_pause(202) > app_appcore_pause
07-06 13:45:48.140+0900 E/cluster-home(  606): homescreen-main.cpp: app_pause(355) >  app pause
07-06 13:45:48.140+0900 E/socket.io(12380): 566: Connected.
07-06 13:45:48.140+0900 E/socket.io(12380): 554: On handshake, sid
07-06 13:45:48.140+0900 E/socket.io(12380): 651: Received Message type(connect)
07-06 13:45:48.140+0900 E/socket.io(12380): 489: On Connected
07-06 13:45:48.150+0900 F/sio_packet(12380): accept()
07-06 13:45:48.150+0900 E/socket.io(12380): 743: encoded paylod length: 18
07-06 13:45:48.160+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:48.270+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:48.270+0900 F/sio_packet(12380): accept()
07-06 13:45:48.270+0900 E/socket.io(12380): 743: encoded paylod length: 21
07-06 13:45:48.270+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:48.310+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:48.310+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:49.670+0900 I/CAPI_APPFW_APPLICATION(11178): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:49.670+0900 W/CAM_APP (11178): cam_sensor_control.c: cam_sensor_rotation_change(166) > [33mignore rotation callback[0m
07-06 13:45:49.690+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:49.740+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:50.320+0900 F/sio_packet(12380): accept()
07-06 13:45:50.320+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:50.320+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.350+0900 F/sio_packet(12380): accept()
07-06 13:45:50.350+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:50.350+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.360+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.360+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.380+0900 F/sio_packet(12380): accept()
07-06 13:45:50.380+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:50.380+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.380+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.400+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.420+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.420+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.430+0900 F/sio_packet(12380): accept()
07-06 13:45:50.430+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:50.430+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.450+0900 F/sio_packet(12380): accept()
07-06 13:45:50.450+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.450+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.450+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.450+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.470+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.470+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.500+0900 F/sio_packet(12380): accept()
07-06 13:45:50.500+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.500+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.510+0900 F/sio_packet(12380): accept()
07-06 13:45:50.510+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.510+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.520+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.520+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.540+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.540+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.550+0900 F/sio_packet(12380): accept()
07-06 13:45:50.550+0900 E/socket.io(12380): 743: encoded paylod length: 75
07-06 13:45:50.550+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.550+0900 I/MALI    (12380): egl_platform_x11_tizen.c: tizen_update_native_surface_private(172) > [EGL-X11] surface->[0xb7d188d8] swap changed from async to sync
07-06 13:45:50.570+0900 F/sio_packet(12380): accept()
07-06 13:45:50.580+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.580+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.580+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.580+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.600+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.600+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.610+0900 F/sio_packet(12380): accept()
07-06 13:45:50.610+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.610+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.670+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.640+0900 F/sio_packet(12380): accept()
07-06 13:45:50.670+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.680+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.680+0900 F/sio_packet(12380): accept()
07-06 13:45:50.680+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.680+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.680+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.710+0900 F/sio_packet(12380): accept()
07-06 13:45:50.710+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.710+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.710+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.710+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.710+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.730+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.740+0900 F/sio_packet(12380): accept()
07-06 13:45:50.740+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.740+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.750+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.750+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.760+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.760+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.770+0900 F/sio_packet(12380): accept()
07-06 13:45:50.770+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.770+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.800+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.800+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.810+0900 F/sio_packet(12380): accept()
07-06 13:45:50.810+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.810+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.820+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.820+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.840+0900 F/sio_packet(12380): accept()
07-06 13:45:50.840+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:50.840+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.860+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.860+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.870+0900 F/sio_packet(12380): accept()
07-06 13:45:50.870+0900 E/socket.io(12380): 743: encoded paylod length: 75
07-06 13:45:50.870+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.900+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.900+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.900+0900 F/sio_packet(12380): accept()
07-06 13:45:50.900+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:50.900+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.920+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.920+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:50.930+0900 F/sio_packet(12380): accept()
07-06 13:45:50.930+0900 E/socket.io(12380): 743: encoded paylod length: 21
07-06 13:45:50.930+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:50.970+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:50.970+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:51.480+0900 I/CAPI_APPFW_APPLICATION(11178): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:51.480+0900 W/CAM_APP (11178): cam_sensor_control.c: cam_sensor_rotation_change(166) > [33mignore rotation callback[0m
07-06 13:45:51.490+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:51.560+0900 I/CAPI_APPFW_APPLICATION(12113): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:52.680+0900 F/sio_packet(12380): accept()
07-06 13:45:52.690+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.690+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.720+0900 F/sio_packet(12380): accept()
07-06 13:45:52.720+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.720+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.730+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.730+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.740+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.740+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.750+0900 F/sio_packet(12380): accept()
07-06 13:45:52.750+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.760+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.770+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.770+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.790+0900 F/sio_packet(12380): accept()
07-06 13:45:52.790+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.790+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.810+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.810+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.810+0900 F/sio_packet(12380): accept()
07-06 13:45:52.810+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.810+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.860+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.860+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.880+0900 F/sio_packet(12380): accept()
07-06 13:45:52.880+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.880+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.900+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.900+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.910+0900 F/sio_packet(12380): accept()
07-06 13:45:52.910+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.910+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.950+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.950+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.960+0900 F/sio_packet(12380): accept()
07-06 13:45:52.960+0900 E/socket.io(12380): 743: encoded paylod length: 75
07-06 13:45:52.960+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:52.970+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:52.970+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:52.980+0900 F/sio_packet(12380): accept()
07-06 13:45:52.980+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:52.990+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.000+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.000+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.020+0900 F/sio_packet(12380): accept()
07-06 13:45:53.020+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.020+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.030+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.030+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.050+0900 F/sio_packet(12380): accept()
07-06 13:45:53.050+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.050+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.070+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.070+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.080+0900 F/sio_packet(12380): accept()
07-06 13:45:53.080+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.080+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.100+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.100+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.120+0900 F/sio_packet(12380): accept()
07-06 13:45:53.120+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.120+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.140+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.140+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.150+0900 F/sio_packet(12380): accept()
07-06 13:45:53.150+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.150+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.160+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.160+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.190+0900 F/sio_packet(12380): accept()
07-06 13:45:53.190+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.190+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.200+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.200+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.220+0900 F/sio_packet(12380): accept()
07-06 13:45:53.220+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.220+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.240+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.240+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.240+0900 F/sio_packet(12380): accept()
07-06 13:45:53.240+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.250+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.260+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.260+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.280+0900 F/sio_packet(12380): accept()
07-06 13:45:53.280+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.280+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.300+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.300+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.310+0900 F/sio_packet(12380): accept()
07-06 13:45:53.310+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.310+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.340+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.340+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.340+0900 F/sio_packet(12380): accept()
07-06 13:45:53.340+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.340+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.360+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.360+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.380+0900 F/sio_packet(12380): accept()
07-06 13:45:53.380+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.380+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.400+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.400+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.410+0900 F/sio_packet(12380): accept()
07-06 13:45:53.410+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.410+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.430+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.430+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.440+0900 F/sio_packet(12380): accept()
07-06 13:45:53.440+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.440+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.460+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.460+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.480+0900 F/sio_packet(12380): accept()
07-06 13:45:53.480+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.480+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.500+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.500+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.510+0900 F/sio_packet(12380): accept()
07-06 13:45:53.510+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.510+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.530+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.530+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.550+0900 F/sio_packet(12380): accept()
07-06 13:45:53.550+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.550+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.570+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.570+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.580+0900 F/sio_packet(12380): accept()
07-06 13:45:53.580+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.580+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.600+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.600+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.620+0900 F/sio_packet(12380): accept()
07-06 13:45:53.620+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.620+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.650+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.650+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.650+0900 F/sio_packet(12380): accept()
07-06 13:45:53.650+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.660+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.670+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.670+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.680+0900 F/sio_packet(12380): accept()
07-06 13:45:53.680+0900 E/socket.io(12380): 743: encoded paylod length: 78
07-06 13:45:53.680+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.700+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.700+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.720+0900 F/sio_packet(12380): accept()
07-06 13:45:53.720+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.720+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.750+0900 F/sio_packet(12380): accept()
07-06 13:45:53.750+0900 E/socket.io(12380): 743: encoded paylod length: 76
07-06 13:45:53.760+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.760+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.760+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.780+0900 F/sio_packet(12380): accept()
07-06 13:45:53.780+0900 E/socket.io(12380): 743: encoded paylod length: 84
07-06 13:45:53.780+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.780+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.780+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.800+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.800+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.810+0900 F/sio_packet(12380): accept()
07-06 13:45:53.820+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:53.820+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.840+0900 F/sio_packet(12380): accept()
07-06 13:45:53.840+0900 E/socket.io(12380): 743: encoded paylod length: 77
07-06 13:45:53.840+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.850+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.850+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.870+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.870+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.880+0900 F/sio_packet(12380): accept()
07-06 13:45:53.880+0900 E/socket.io(12380): 743: encoded paylod length: 80
07-06 13:45:53.880+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.900+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.900+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.910+0900 F/sio_packet(12380): accept()
07-06 13:45:53.910+0900 E/socket.io(12380): 743: encoded paylod length: 81
07-06 13:45:53.910+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.930+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.930+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.940+0900 F/sio_packet(12380): accept()
07-06 13:45:53.940+0900 E/socket.io(12380): 743: encoded paylod length: 80
07-06 13:45:53.940+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.960+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.960+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:53.980+0900 F/sio_packet(12380): accept()
07-06 13:45:53.980+0900 E/socket.io(12380): 743: encoded paylod length: 80
07-06 13:45:53.980+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:53.990+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:53.990+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:54.000+0900 F/sio_packet(12380): accept()
07-06 13:45:54.000+0900 E/socket.io(12380): 743: encoded paylod length: 80
07-06 13:45:54.000+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:54.020+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:54.020+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:54.020+0900 F/sio_packet(12380): accept()
07-06 13:45:54.020+0900 E/socket.io(12380): 743: encoded paylod length: 21
07-06 13:45:54.030+0900 E/socket.io(12380): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 13:45:54.070+0900 E/socket.io(12380): 669: Received Message type(Event)
07-06 13:45:54.070+0900 F/get_binary(12380): in get binary_message()...
07-06 13:45:55.850+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: app_efl_exit(145) > app_efl_exit
07-06 13:45:55.850+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 22
07-06 13:45:55.850+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(803) > app status : 5
07-06 13:45:55.850+0900 I/APP_CORE(12380): appcore-efl.c: __after_loop(1057) > Legacy lifecycle: 0
07-06 13:45:55.850+0900 I/APP_CORE(12380): appcore-efl.c: __after_loop(1059) > [APP 12380] PAUSE before termination
07-06 13:45:55.850+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_pause(607) > app_appcore_pause
07-06 13:45:55.850+0900 I/CAPI_APPFW_APPLICATION(12380): app_main.c: _ui_app_appcore_terminate(581) > app_appcore_terminate
07-06 13:45:56.200+0900 W/AUL_AMD (  452): amd_key.c: _key_ungrab(250) > fail(-1) to ungrab key(XF86Stop)
07-06 13:45:56.200+0900 W/AUL_AMD (  452): amd_launch.c: __e17_status_handler(2132) > back key ungrab error
07-06 13:45:56.210+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(142) > dead_pid = 12380 pgid = 12380
07-06 13:45:56.210+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(123) > dead_pid(12380)
07-06 13:45:56.210+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
07-06 13:45:56.210+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
07-06 13:45:56.210+0900 I/Tizen::System( 1047): (246) > Terminated app [org.tizen.other]
07-06 13:45:56.210+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 13:45:56.210+0900 I/AUL_AMD (  452): amd_main.c: __app_dead_handler(256) > __app_dead_handler, pid: 12380
07-06 13:45:56.210+0900 I/Tizen::App( 1034): (243) > App[org.tizen.other] pid[12380] terminate event is forwarded
07-06 13:45:56.210+0900 I/Tizen::System( 1034): (256) > osp.accessorymanager.service provider is found.
07-06 13:45:56.210+0900 I/Tizen::System( 1034): (196) > Accessory Owner is removed [org.tizen.other, 12380, ]
07-06 13:45:56.210+0900 I/Tizen::System( 1034): (256) > osp.system.service provider is found.
07-06 13:45:56.210+0900 I/Tizen::App( 1034): (506) > TerminatedApp(org.tizen.other)
07-06 13:45:56.210+0900 I/Tizen::App( 1034): (512) > Not registered pid(12380)
07-06 13:45:56.210+0900 I/Tizen::App( 1034): (782) > Finished invoking application event listener for org.tizen.other, 12380.
07-06 13:45:56.220+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 13:45:56.220+0900 I/Tizen::App( 1047): (782) > Finished invoking application event listener for org.tizen.other, 12380.
07-06 13:45:56.240+0900 I/CAPI_APPFW_APPLICATION(11178): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 13:45:56.240+0900 W/CAM_APP (11178): cam_sensor_control.c: cam_sensor_rotation_change(166) > [33mignore rotation callback[0m
07-06 13:45:56.270+0900 W/CRASH_MANAGER(12410): worker.c: worker_job(1236) > 11123806f7468143615795
