S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 2496
Date: 2015-04-13 18:59:01+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 2
      invalid permissions for mapped object
      si_addr = 0xb6ef313c

Register Information
r0   = 0x00000000, r1   = 0x00000000
r2   = 0xcd530c00, r3   = 0xaf197008
r4   = 0xb6ef313c, r5   = 0xb0f67f6c
r6   = 0xb0f67f70, r7   = 0xb0f67f74
r8   = 0xb6fe2668, r9   = 0xb71409d4
r10  = 0x000000ff, fp   = 0x000000d8
ip   = 0xb6f19ae4, sp   = 0xb0f67f08
lr   = 0xb3dc7a7b, pc   = 0xb3ded29e
cpsr = 0x60000030

Memory Information
MemTotal:   797840 KB
MemFree:    324956 KB
Buffers:     79504 KB
Cached:     169548 KB
VmPeak:     156864 KB
VmSize:     129864 KB
VmLck:           0 KB
VmHWM:       26396 KB
VmRSS:       26396 KB
VmData:      73976 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24984 KB
VmPTE:          96 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 2496 TID = 2506
2496 2500 2501 2502 2503 2504 2506 2507 

Maps Information
b17aa000 b17ab000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17e8000 b17e9000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17f1000 b17f8000 r-xp /usr/lib/libfeedback.so.0.1.4
b180b000 b180c000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b1814000 b182b000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b19b1000 b19b6000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31ff000 b324a000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b3253000 b325d000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3266000 b32c2000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3acf000 b3ad5000 r-xp /usr/lib/libUMP.so
b3add000 b3af0000 r-xp /usr/lib/libEGL_platform.so
b3af9000 b3bd0000 r-xp /usr/lib/libMali.so
b3bdb000 b3bf2000 r-xp /usr/lib/libEGL.so.1.4
b3bfb000 b3c00000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c09000 b3c0a000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c13000 b3c2b000 r-xp /usr/lib/libpng12.so.0.50.0
b3c33000 b3c71000 r-xp /usr/lib/libGLESv2.so.2.0
b3c79000 b3c7d000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c86000 b3c88000 r-xp /usr/lib/libdri2.so.0.0.0
b3c90000 b3c97000 r-xp /usr/lib/libdrm.so.2.4.0
b3ca0000 b3d56000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d61000 b3d77000 r-xp /usr/lib/libtts.so
b3d80000 b3d87000 r-xp /usr/lib/libtbm.so.1.0.0
b3d8f000 b3d94000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3d9c000 b3dad000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3db5000 b3dbc000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3dc4000 b3dc9000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3dd1000 b3dd3000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3ddb000 b3de3000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3deb000 b3dee000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3df7000 b3eb1000 r-xp /opt/usr/apps/org.tizen.other/bin/other
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
b6f9c000 b7246000 rw-p [heap]
bec46000 bec67000 rwxp [stack]
End of Maps Information

Callstack Information (PID:2496)
Call Stack Count: 70
 0: image_util_decode_jpeg_from_memory + 0x3d (0xb3ded29e) [/usr/lib/libcapi-media-image-util.so.0] + 0x229e
 1: setTextureData + 0x122 (0xb3dffd23) [/opt/usr/apps/org.tizen.other/bin/other] + 0x8d23
 2: socket_io_client::{lambda(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)#1}::operator()(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&) const + 0x1fe (0xb3e79453) [/opt/usr/apps/org.tizen.other/bin/other] + 0x82453
 3: _ZNSt17_Function_handlerIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS5_EZ16socket_io_clientEUlS1_S7_bS8_E_E9_M_invokeERKSt9_Any_dat + 0x4a (0xb3e79a8b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x82a8b
 4: std::function<void (std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)>::operator()(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&) const + 0x5a (0xb3e1a9fb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x239fb
 5: sio::event_adapter::adapt_func(std::function<void (std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)> const&, sio::event&) + 0x3a (0xb3e15d3b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1ed3b
 6: _ZNSt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS6_EERNS4_5eventEESB_St12_PlaceholderILi1EEEE6__callIvISF_ + 0x4e (0xb3e30b1b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x39b1b
 7: _ZNSt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS6_EERNS4_5eventEESB_St12_PlaceholderILi1EEEEclIISF_EvEET0 + 0x2c (0xb3e290d1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x320d1
 8: _ZNSt17_Function_handlerIFvRN3sio5eventEESt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrINS0_7messageEEbRSA_EES2_ESF_St12_Plac + 0x22 (0xb3e2132b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2a32b
 9: std::function<void (sio::event&)>::operator()(sio::event&) const + 0x30 (0xb3e1d60d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2660d
10: sio::client::impl::on_socketio_event(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int, std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&) + 0x96 (0xb3e04f7b) [/opt/usr/apps/org.tizen.other/bin/other] + 0xdf7b
11: sio::client::impl::on_decode(sio::packet const&) + 0x264 (0xb3e03529) [/opt/usr/apps/org.tizen.other/bin/other] + 0xc529
12: void std::_Mem_fn<void (sio::client::impl::*)(sio::packet const&)>::operator()<sio::packet const&, void>(sio::client::impl*, sio::packet const&) const + 0x4a (0xb3e38567) [/opt/usr/apps/org.tizen.other/bin/other] + 0x41567
13: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvRKNS1_6packetEEEPS3_St12_PlaceholderILi1EEEE6__callIvIS6_EILj0ELj1EEEET_OSt5tupleII + 0x52 (0xb3e316d7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3a6d7
14: void std::_Bind<std::_Mem_fn<void (sio::client::impl::*)(sio::packet const&)> (sio::client::impl*, std::_Placeholder<1>)>::operator()<sio::packet const&, void>(sio::packet const&) + 0x2c (0xb3e29f31) [/opt/usr/apps/org.tizen.other/bin/other] + 0x32f31
15: _ZNSt17_Function_handlerIFvRKN3sio6packetEESt5_BindIFSt7_Mem_fnIMNS0_6client4implEFvS3_EEPS8_St12_PlaceholderILi1EEEEE9_M_invoke + 0x22 (0xb3e22263) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2b263
16: std::function<void (sio::packet const&)>::operator()(sio::packet const&) const + 0x30 (0xb3e71d2d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7ad2d
17: sio::packet_manager::put_payload(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x11a (0xb3e6dfff) [/opt/usr/apps/org.tizen.other/bin/other] + 0x76fff
18: _ZN3sio6client4impl10on_messageESt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS6_5alloc15con_msg_managerE + 0xd4 (0xb3e031f5) [/opt/usr/apps/org.tizen.other/bin/other] + 0xc1f5
19: _ZNKSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS7_5alloc15con_msg_mana + 0x76 (0xb3e38473) [/opt/usr/apps/org.tizen.other/bin/other] + 0x41473
20: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS8_5alloc15con + 0x6e (0xb3e3158b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3a58b
21: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS8_5alloc15con + 0x38 (0xb3e29d2d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x32d2d
22: _ZNSt17_Function_handlerIFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS4_5alloc15con_msg_managerEEEEES + 0x2e (0xb3e2205f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2b05f
23: std::function<void (std::weak_ptr<void>, std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager> >)>::operator()(std::weak_ptr<void>, std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager> >) const + 0x5c (0xb3e507f1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x597f1
24: websocketpp::connection<websocketpp::config::debug_asio>::handle_read_frame(std::error_code const&, unsigned int) + 0x6d0 (0xb3e4ccc9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x55cc9
25: void std::_Mem_fn<void (websocketpp::connection<websocketpp::config::debug_asio>::*)(std::error_code const&, unsigned int)>::operator()<std::error_code const&, unsigned int, void>(websocketpp::connection<websocketpp::config::debug_asio>*, std::error_code const&, unsigned int&&) const + 0x58 (0xb3e62de1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6bde1
26: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS1_6config10debug_asioEEEFvRKSt10error_codejEEPS5_St12_PlaceholderILi1EESD_ + 0x6e (0xb3e6214f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6b14f
27: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS1_6config10debug_asioEEEFvRKSt10error_codejEEPS5_St12_PlaceholderILi1EESD_ + 0x38 (0xb3e61199) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6a199
28: _ZNSt17_Function_handlerIFvRKSt10error_codejESt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS6_6config10debug_asioEEEFvS2_jEE + 0x30 (0xb3e601b9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x691b9
29: std::function<void (std::error_code const&, unsigned int)>::operator()(std::error_code const&, unsigned int) const + 0x40 (0xb3e43f95) [/opt/usr/apps/org.tizen.other/bin/other] + 0x4cf95
30: _ZN11websocketpp9transport4asio10connectionINS_6config10debug_asio16transport_configEE17handle_async_readERKN5boost6system10erro + 0x126 (0xb3e3ee87) [/opt/usr/apps/org.tizen.other/bin/other] + 0x47e87
31: _ZNKSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS0_6config10debug_asio16transport_configEEEFvRKN5boost6system10error_ + 0x5e (0xb3e624ff) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6b4ff
32: _ZNKSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS0_6config10debug_asio16transport_configEEEFvRKN5boost6system10error_ + 0x34 (0xb3e61785) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6a785
33: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS1_6config10debug_asio16transport_configEEEFvRKN5boost6syste + 0x6c (0xb3e60665) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69665
34: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS1_6config10debug_asio16transport_configEEEFvRKN5boost6syste + 0x38 (0xb3e5f4d5) [/opt/usr/apps/org.tizen.other/bin/other] + 0x684d5
35: _ZN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS5_6config10debug_asio16transport + 0x1e (0xb3e5e0eb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x670eb
36: _ZN5boost4asio19asio_handler_invokeINS0_6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6con + 0xc (0xb3e5ca51) [/opt/usr/apps/org.tizen.other/bin/other] + 0x65a51
37: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10c + 0x1a (0xb3e5b6cf) [/opt/usr/apps/org.tizen.other/bin/other] + 0x646cf
38: _ZN5boost4asio6detail19asio_handler_invokeINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6con + 0x14 (0xb3e59aa9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x62aa9
39: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10c + 0x1a (0xb3e57403) [/opt/usr/apps/org.tizen.other/bin/other] + 0x60403
40: _ZN5boost4asio6detail18completion_handlerINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6conf + 0x72 (0xb3e5750b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6050b
41: _ZN5boost4asio6detail14strand_service8dispatchINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS7_ + 0xf4 (0xb3e54d85) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5dd85
42: _ZN5boost4asio10io_service6strand8dispatchINS0_6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionIN + 0x2c (0xb3e51fd1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5afd1
43: _ZN5boost4asio6detail15wrapped_handlerINS0_10io_service6strandESt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionIN + 0x3e (0xb3e4e333) [/opt/usr/apps/org.tizen.other/bin/other] + 0x57333
44: _ZNSt17_Function_handlerIFvRKN5boost6system10error_codeEjENS0_4asio6detail15wrapped_handlerINS6_10io_service6strandESt5_BindIFSt + 0x30 (0xb3e497e5) [/opt/usr/apps/org.tizen.other/bin/other] + 0x527e5
45: std::function<void (boost::system::error_code const&, unsigned int)>::operator()(boost::system::error_code const&, unsigned int) const + 0x40 (0xb3e4de31) [/opt/usr/apps/org.tizen.other/bin/other] + 0x56e31
46: void websocketpp::transport::asio::custom_alloc_handler<std::function<void (boost::system::error_code const&, unsigned int)> >::operator()<boost::system::error_code, unsigned int>(boost::system::error_code, unsigned int) + 0x20 (0xb3e49385) [/opt/usr/apps/org.tizen.other/bin/other] + 0x52385
47: _ZN5boost4asio6detail7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_serviceIS5_EEEENS0_17mutable_buffers_1EN + 0xe0 (0xb3e4b2d5) [/opt/usr/apps/org.tizen.other/bin/other] + 0x542d5
48: _ZN5boost4asio6detail7binder2INS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_serviceIS6_EEEENS0_17mutabl + 0x22 (0xb3e60b73) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69b73
49: _ZN5boost4asio19asio_handler_invokeINS0_6detail7binder2INS2_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_s + 0xc (0xb3e5f969) [/opt/usr/apps/org.tizen.other/bin/other] + 0x68969
50: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2INS3_7read_opINS2_19basic_stream_socketINS2_2ip3tcpENS2 + 0x1a (0xb3e5e69f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6769f
51: _ZN5boost4asio6detail19asio_handler_invokeINS1_7binder2INS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_s + 0x16 (0xb3e5d24b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6624b
52: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2INS3_7read_opINS2_19basic_stream_socketINS2_2ip3tcpENS2 + 0x1a (0xb3e5c1eb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x651eb
53: _ZN5boost4asio6detail23reactive_socket_recv_opINS0_17mutable_buffers_1ENS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21 + 0x7a (0xb3e5acb3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x63cb3
54: boost::asio::detail::task_io_service_operation::complete(boost::asio::detail::task_io_service&, boost::system::error_code const&, unsigned int) + 0x1c (0xb3e0fa61) [/opt/usr/apps/org.tizen.other/bin/other] + 0x18a61
55: _ZN5boost4asio6detail13epoll_reactor16descriptor_state11do_completeEPNS1_15task_io_serviceEPNS1_25task_io_service_operationERKNS + 0x3a (0xb3e1119b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1a19b
56: boost::asio::detail::task_io_service_operation::complete(boost::asio::detail::task_io_service&, boost::system::error_code const&, unsigned int) + 0x1c (0xb3e0fa61) [/opt/usr/apps/org.tizen.other/bin/other] + 0x18a61
57: _ZN5boost4asio6detail15task_io_service10do_run_oneERNS1_11scoped_lockINS1_11posix_mutexEEERNS1_27task_io_service_thread_infoERKN + 0x116 (0xb3e11a13) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1aa13
58: boost::asio::detail::task_io_service::run(boost::system::error_code&) + 0xca (0xb3e115e3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1a5e3
59: boost::asio::io_service::run() + 0x22 (0xb3e11c8b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1ac8b
60: websocketpp::transport::asio::endpoint<websocketpp::config::debug_asio::transport_config>::run() + 0x12 (0xb3e1d483) [/opt/usr/apps/org.tizen.other/bin/other] + 0x26483
61: sio::client::impl::run_loop() + 0x12 (0xb3e04eb3) [/opt/usr/apps/org.tizen.other/bin/other] + 0xdeb3
62: _ZNKSt7_Mem_fnIMN3sio6client4implEFvvEEclIJEvEEvPS2_DpOT_ + 0x3e (0xb3e6bf2f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x74f2f
63: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS3_EE6__callIvJEJLj0EEEET_OSt5tupleIJDpT0_EESt12_Index_tupleIJXspT1_EEE + 0x34 (0xb3e6bc81) [/opt/usr/apps/org.tizen.other/bin/other] + 0x74c81
64: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS3_EEclIJEvEET0_DpOT_ + 0x1a (0xb3e6b99f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7499f
65: _ZNSt12_Bind_simpleIFSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS4_EEvEE9_M_invokeIJEEEvSt12_Index_tupleIJXspT_EEE + 0x22 (0xb3e6b097) [/opt/usr/apps/org.tizen.other/bin/other] + 0x74097
66: std::_Bind_simple<std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)> ()>::operator()() + 0x10 (0xb3e6a531) [/opt/usr/apps/org.tizen.other/bin/other] + 0x73531
67: std::thread::_Impl<std::_Bind_simple<std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)> ()> >::_M_run() + 0x12 (0xb3e68c13) [/opt/usr/apps/org.tizen.other/bin/other] + 0x71c13
68: (0xb5b8c7dd) [/usr/lib/libstdc++.so.6] + 0x517dd
69: (0xb6ee7d30) [/lib/libpthread.so.0] + 0x6d30
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
0
04-13 18:58:50.305+0900 D/PKGMGR  ( 2245): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.305+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:60] for org.tizen.other
04-13 18:58:50.305+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 60
04-13 18:58:50.305+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.305+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2198): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.310+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.310+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(419) > event_cb is called
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.310+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2400): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.310+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.310+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2276): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.310+0900 D/DATA_PROVIDER_MASTER( 2276): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.310+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.310+0900 D/QUICKPANEL( 2279): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 60
04-13 18:58:50.310+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.310+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 D/PKGMGR  ( 2274): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[60]
04-13 18:58:50.310+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.310+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.310+0900 E/PKGMGR_CERT( 2431): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 2 14
04-13 18:58:50.315+0900 E/PKGMGR_CERT( 2431): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 10 3
04-13 18:58:50.315+0900 E/PKGMGR_CERT( 2431): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 4 5
04-13 18:58:50.315+0900 E/PKGMGR_CERT( 2431): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 5 5
04-13 18:58:50.315+0900 E/PKGMGR_CERT( 2431): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(496) > Id:Count = 6 5
04-13 18:58:50.330+0900 E/PKGMGR_CERT( 2431): pkgmgrinfo_certinfo.c: pkgmgrinfo_save_certinfo(571) > Transaction Commit and End
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/res]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_make_directory(1490) > mkdir failed. appdir=[/usr/apps/org.tizen.other/shared], errno=[2][No such file or directory]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/opt/usr/apps/org.tizen.other/shared/data]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_directory_policy(1123) > skip! empty dirpath=[/usr/apps/org.tizen.other/shared/res]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/tizen-manifest.xml]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/author-signature.xml]
04-13 18:58:50.335+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/apps/org.tizen.other/signature1.xml]
04-13 18:58:50.340+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_file_policy(1108) > skip! empty filepath=[/usr/share/packages/org.tizen.other.xml]
04-13 18:58:50.340+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_make_directory_for_ext(1353) > Directory dose not exist. path: /opt/usr/apps/org.tizen.other/shared/data, errno: 2 (No such file or directory)
04-13 18:58:50.460+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_register_package(64) > [smack] app_install(org.tizen.other), result=[0]
04-13 18:58:50.460+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other, 5, _), result=[0]
04-13 18:58:50.465+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared, 5, _), result=[0]
04-13 18:58:50.465+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/res, 5, _), result=[0]
04-13 18:58:50.465+0900 E/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_get_smack_label_access(627) > Error in getting smack ACCESS label failed. result[-1] (path:[/opt/usr/apps/org.tizen.other/shared/data]))
04-13 18:58:50.465+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_get_group_id(1770) > encoding done, len=[28]
04-13 18:58:50.465+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_apply_smack(1878) > groupid = [QVI9+j1mlN94KHCZf1TZSIDuLJk=] for shared/trusted.
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/shared/trusted, 1, QVI9+j1mlN94KHCZf1TZSIDuLJk=), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/bin, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/lib, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/tizen-manifest.xml, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/author-signature.xml, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/usr/apps/org.tizen.other/signature1.xml, 0, org.tizen.other), result=[0]
04-13 18:58:50.530+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/share/packages/org.tizen.other.xml, 0, org.tizen.other), result=[0]
04-13 18:58:50.535+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other, 5, _), result=[0]
04-13 18:58:50.535+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/data, 0, org.tizen.other), result=[0]
04-13 18:58:50.535+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/cache, 0, org.tizen.other), result=[0]
04-13 18:58:50.535+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_setup_path(113) > [smack] app_setup_path(org.tizen.other, /opt/storage/sdcard/apps/org.tizen.other/shared, 5, _), result=[0]
04-13 18:58:50.540+0900 D/rpm-installer( 2431): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/internet]
04-13 18:58:50.555+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-13 18:58:50.560+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-13 18:58:50.650+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-13 18:58:50.650+0900 D/rpm-installer( 2431): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-13 18:58:50.650+0900 D/rpm-installer( 2431): rpm-installer.c: __privilege_func(1086) > package_id = [org.tizen.other], privilege = [http://tizen.org/privilege/systemsettings]
04-13 18:58:50.670+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-13 18:58:50.670+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-13 18:58:50.760+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-13 18:58:50.760+0900 D/rpm-installer( 2431): rpm-installer.c: __privilege_func(1112) > _ri_privilege_enable_permissions(org.tizen.other, 7) succeed.
04-13 18:58:50.785+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: __ri_privilege_perm_begin(44) > [smack] perm_begin, result=[0]
04-13 18:58:50.785+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: _ri_privilege_enable_permissions(97) > [smack] app_enable_permissions(org.tizen.other, 7), result=[0]
04-13 18:58:50.890+0900 D/rpm-installer( 2431): rpm-installer-privilege.c: __ri_privilege_perm_end(54) > [smack] perm_end, result=[0]
04-13 18:58:50.890+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_package_install(2194) > permission applying done successfully.
04-13 18:58:50.890+0900 D/PRIVILEGE_INFO( 2431): privilege_info.c: privilege_manager_verify_privilege_list(625) > privilege_info_compare_privilege_level called
04-13 18:58:50.890+0900 D/PRIVILEGE_INFO( 2431): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/internet
04-13 18:58:50.895+0900 D/PRIVILEGE_INFO( 2431): privilege_info.c: privilege_manager_verify_privilege_list(641) > Checking privilege : http://tizen.org/privilege/systemsettings
04-13 18:58:50.900+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_verify_privilege_list(672) > privilege_manager_verify_privilege_list(PRVMGR_PACKAGE_TYPE_CORE) is ok.
04-13 18:58:50.900+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_package_install(2202) > _coretpk_installer_verify_privilege_list done.
04-13 18:58:50.900+0900 D/rpm-installer( 2431): rpm-vconf-intf.c: _ri_broadcast_status_notification(188) > pkgid=[org.tizen.other], key=[install_percent], val=[100]
04-13 18:58:50.900+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_package_install(2224) > install status is [2].
04-13 18:58:50.900+0900 D/rpm-installer( 2431): coretpk-installer.c: __post_install_for_mmc(311) > Installed storage is internal.
04-13 18:58:50.900+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_package_install(2231) > _coretpk_installer_package_install is done.
04-13 18:58:50.900+0900 D/rpm-installer( 2431): rpm-vconf-intf.c: _ri_broadcast_status_notification(196) > pkgid=[org.tizen.other], key=[end], val=[ok]
04-13 18:58:50.900+0900 D/rpm-installer( 2431): coretpk-installer.c: _coretpk_installer_prepare_package_install(2695) > success
04-13 18:58:50.900+0900 D/rpm-installer( 2431): rpm-appcore-intf.c: main(224) > sync() start
04-13 18:58:50.905+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2198): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.905+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/QUICKPANEL( 2279): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:install_percent val:100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-13 18:58:50.905+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2276): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/DATA_PROVIDER_MASTER( 2276): pkgmgr.c: progress_cb(374) > [SECURE_LOG] [org.tizen.other] 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(419) > event_cb is called
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.905+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.905+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2245): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [install_percent:100] for org.tizen.other
04-13 18:58:50.905+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _install_percent(440) > package(org.tizen.other) with 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.905+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2400): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.905+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / install_percent / 100
04-13 18:58:50.905+0900 D/PKGMGR  ( 2274): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[install_percent] val[100]
04-13 18:58:50.905+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.905+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(396) > [SECURE_LOG] __operation_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): pkgmgr.c: __find_op_cbinfo(328) > tmp->req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191, req_key /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(405) > __find_op_cbinfo
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): pkgmgr.c: __operation_callback(419) > event_cb is called
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.910+0900 D/PKGMGR  ( 2429): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.910+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.910+0900 D/PKGMGR  ( 2400): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.910+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.910+0900 D/PKGMGR  ( 2400): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.910+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.910+0900 D/PKGMGR  ( 2279): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.910+0900 D/QUICKPANEL( 2279): uninstall.c: _pkgmgr_event_cb(81) > [SECURE_LOG] [_pkgmgr_event_cb : 81] pkg:org.tizen.other key:end val:ok
04-13 18:58:50.910+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.910+0900 D/PKGMGR  ( 2279): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.910+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.910+0900 D/PKGMGR  ( 2245): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.910+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _pkgmgr_cb(580) > pkgmgr request [end:ok] for org.tizen.other
04-13 18:58:50.910+0900 D/MENU_SCREEN( 2245): pkgmgr.c: _end(520) > Package(org.tizen.other) : key(install) - val(ok)
04-13 18:58:50.915+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.915+0900 D/PKGMGR  ( 2276): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.915+0900 D/DATA_PROVIDER_MASTER( 2276): pkgmgr.c: end_cb(409) > [SECURE_LOG] [org.tizen.other] ok
04-13 18:58:50.915+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.915+0900 D/PKGMGR  ( 2198): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.915+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.915+0900 D/PKGMGR  ( 2198): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.915+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [status] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.915+0900 D/PKGMGR  ( 2274): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.915+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.915+0900 D/PKGMGR  ( 2274): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.915+0900 D/PKGMGR  ( 2170): comm_client_gdbus.c: _on_signal_handle_filter(183) > [SECURE_LOG] Got signal: [install] /opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191 / coretpk / org.tizen.other / end / ok
04-13 18:58:50.915+0900 D/PKGMGR  ( 2170): pkgmgr.c: __status_callback(440) > [SECURE_LOG] __status_callback() req_id[/opt/usr/apps/tmp/org.tizen.other-1.0.0-arm.tpk_689006191] pkg_type[coretpk] pkgid[org.tizen.other]key[end] val[ok]
04-13 18:58:50.915+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(538) > [SECURE_LOG] pkgid(org.tizen.other), key(end), value(ok)
04-13 18:58:50.915+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __amd_pkgmgrinfo_status_cb(559) > [SECURE_LOG] op(install), value(ok)
04-13 18:58:50.945+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.945+0900 D/PKGMGR  ( 2276): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.965+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __app_info_insert_handler(185) > __app_info_insert_handler
04-13 18:58:50.965+0900 D/AUL_AMD ( 2170): amd_appinfo.c: __app_info_insert_handler(388) > [SECURE_LOG] appinfo file:org.tizen.other, comp:ui, type:rpm
04-13 18:58:50.965+0900 D/PKGMGR  ( 2170): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:50.965+0900 D/PKGMGR  ( 2170): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:50.980+0900 D/MENU_SCREEN( 2245): layout.c: layout_create_package(200) > package org.tizen.other is installed directly
04-13 18:58:51.290+0900 D/MENU_SCREEN( 2245): item.c: item_update(594) > Access to file [/opt/usr/apps/org.tizen.other/shared/res/other.png], size[57662]
04-13 18:58:51.315+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
04-13 18:58:51.320+0900 D/rpm-installer( 2431): rpm-appcore-intf.c: main(226) > sync() end
04-13 18:58:51.325+0900 D/rpm-installer( 2431): rpm-appcore-intf.c: main(245) > ------------------------------------------------
04-13 18:58:51.325+0900 D/rpm-installer( 2431): rpm-appcore-intf.c: main(246) >  [END] rpm-installer: result=[0]
04-13 18:58:51.325+0900 D/rpm-installer( 2431): rpm-appcore-intf.c: main(247) > ------------------------------------------------
04-13 18:58:51.355+0900 D/PKGMGR_SERVER( 2419): pkgmgr-server.c: sighandler(326) > child exit [2431]
04-13 18:58:51.360+0900 D/PKGMGR_SERVER( 2419): pkgmgr-server.c: sighandler(341) > child NORMAL exit [2431]
04-13 18:58:51.395+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(189) > callback function is end
04-13 18:58:51.395+0900 D/PKGMGR  ( 2245): comm_client_gdbus.c: _on_signal_handle_filter(191) > Handled signal. Exit function
04-13 18:58:52.185+0900 D/PKGMGR_SERVER( 2419): pkgmgr-server.c: exit_server(724) > exit_server Start
04-13 18:58:52.185+0900 D/PKGMGR_SERVER( 2419): pkgmgr-server.c: main(1516) > Quit main loop.
04-13 18:58:52.185+0900 D/PKGMGR_SERVER( 2419): pkgmgr-server.c: main(1524) > package manager server terminated.
04-13 18:58:58.525+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 0
04-13 18:58:58.530+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-13 18:58:58.545+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 2493, pid = 2495
04-13 18:58:58.550+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-13 18:58:58.555+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
04-13 18:58:58.555+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-13 18:58:58.555+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-13 18:58:58.555+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-13 18:58:58.555+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-13 18:58:58.555+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-13 18:58:58.555+0900 D/AUL_PAD ( 2204): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-13 18:58:58.560+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 2496 /opt/usr/apps/org.tizen.other/bin/other
04-13 18:58:58.560+0900 D/AUL_PAD ( 2496): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-13 18:58:58.560+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-13 18:58:58.560+0900 D/AUL_PAD ( 2496): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-13 18:58:58.560+0900 D/AUL_PAD ( 2496): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-13 18:58:58.580+0900 D/AUL_PAD ( 2496): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-13 18:58:58.580+0900 D/AUL_PAD ( 2496): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-13 18:58:58.580+0900 D/AUL_PAD ( 2496): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-13 18:58:58.580+0900 D/AUL_PAD ( 2496): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-13 18:58:58.580+0900 D/LAUNCH  ( 2496): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-13 18:58:58.620+0900 I/CAPI_APPFW_APPLICATION( 2496): app_main.c: ui_app_main(697) > app_efl_main
04-13 18:58:58.620+0900 D/LAUNCH  ( 2496): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-13 18:58:58.660+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-13 18:58:58.660+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-13 18:58:58.660+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-13 18:58:58.660+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-13 18:58:58.660+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-13 18:58:58.660+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 2496
04-13 18:58:58.660+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-13 18:58:58.660+0900 D/RESOURCED( 2372): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-13 18:58:58.660+0900 D/RESOURCED( 2372): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
04-13 18:58:58.660+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 2496, type 4 
04-13 18:58:58.660+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 2496
04-13 18:58:58.660+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-13 18:58:58.660+0900 D/RESOURCED( 2372): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 2496
04-13 18:58:58.660+0900 D/RESOURCED( 2372): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 2496
04-13 18:58:58.665+0900 D/RESOURCED( 2372): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-13 18:58:58.665+0900 D/RESOURCED( 2372): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-13 18:58:58.665+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 18:58:58.665+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 18:58:58.665+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 18:58:58.665+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 18:58:58.665+0900 E/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 509
04-13 18:58:58.665+0900 D/RESOURCED( 2372): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-13 18:58:58.675+0900 D/APP_CORE( 2496): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-13 18:58:58.690+0900 D/AUL     ( 2496): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 2496 is org.tizen.other
04-13 18:58:58.690+0900 D/APP_CORE( 2496): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-13 18:58:58.690+0900 D/APP_CORE( 2496): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-13 18:58:58.690+0900 D/AUL     ( 2496): app_sock.c: __create_server_sock(135) > pg path - already exists
04-13 18:58:58.690+0900 D/LAUNCH  ( 2496): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-13 18:58:58.690+0900 I/CAPI_APPFW_APPLICATION( 2496): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-13 18:58:58.875+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2496
04-13 18:58:58.925+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-13 18:58:58.930+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
04-13 18:58:58.930+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-13 18:58:58.935+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
04-13 18:58:58.935+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-13 18:58:58.940+0900 F/socket.io( 2496): thread_start
04-13 18:58:58.940+0900 F/socket.io( 2496): finish 0
04-13 18:58:58.940+0900 D/LAUNCH  ( 2496): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-13 18:58:58.940+0900 D/APP_CORE( 2496): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-13 18:58:58.940+0900 D/APP_CORE( 2496): appcore.c: __aul_handler(423) > [APP 2496]     AUL event: AUL_START
04-13 18:58:58.940+0900 D/APP_CORE( 2496): appcore-efl.c: __do_app(470) > [APP 2496] Event: RESET State: CREATED
04-13 18:58:58.940+0900 D/APP_CORE( 2496): appcore-efl.c: __do_app(496) > [APP 2496] RESET
04-13 18:58:58.940+0900 D/LAUNCH  ( 2496): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-13 18:58:58.940+0900 I/CAPI_APPFW_APPLICATION( 2496): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-13 18:58:58.940+0900 D/APP_SVC ( 2496): appsvc.c: __set_bundle(161) > __set_bundle
04-13 18:58:58.940+0900 D/LAUNCH  ( 2496): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-13 18:58:58.955+0900 I/APP_CORE( 2496): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-13 18:58:58.955+0900 I/APP_CORE( 2496): appcore-efl.c: __do_app(509) > [APP 2496] Initial Launching, call the resume_cb
04-13 18:58:58.955+0900 I/CAPI_APPFW_APPLICATION( 2496): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-13 18:58:58.955+0900 D/APP_CORE( 2496): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-13 18:58:58.960+0900 D/APP_CORE( 2496): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600003
04-13 18:58:58.960+0900 D/APP_CORE( 2496): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600003
04-13 18:58:59.135+0900 D/APP_CORE( 2245): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-13 18:58:59.135+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-13 18:58:59.135+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-13 18:58:59.135+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(470) > [APP 2245] Event: PAUSE State: RUNNING
04-13 18:58:59.135+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(538) > [APP 2245] PAUSE
04-13 18:58:59.135+0900 I/CAPI_APPFW_APPLICATION( 2245): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-13 18:58:59.135+0900 D/MENU_SCREEN( 2245): menu_screen.c: _pause_cb(538) > Pause start
04-13 18:58:59.135+0900 D/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 18:58:59.135+0900 E/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 18:58:59.135+0900 D/APP_CORE( 2496): appcore.c: __prt_ltime(183) > [APP 2496] first idle after reset: 616 msec
04-13 18:58:59.135+0900 D/APP_CORE( 2496): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 0
04-13 18:58:59.135+0900 D/APP_CORE( 2496): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-13 18:58:59.135+0900 D/APP_CORE( 2496): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-13 18:58:59.140+0900 D/APP_CORE( 2496): appcore-efl.c: __do_app(470) > [APP 2496] Event: RESUME State: RUNNING
04-13 18:58:59.140+0900 D/LAUNCH  ( 2496): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-13 18:58:59.140+0900 D/LAUNCH  ( 2496): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-13 18:58:59.140+0900 D/LAUNCH  ( 2496): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-13 18:58:59.140+0900 D/APP_CORE( 2496): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 18:58:59.140+0900 E/APP_CORE( 2496): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 18:58:59.140+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2245, type = 2
04-13 18:58:59.140+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2496, type = 0
04-13 18:58:59.140+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(2496) status(3)
04-13 18:58:59.140+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 2496
04-13 18:58:59.140+0900 I/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 2496, oom : 200
04-13 18:58:59.140+0900 E/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-13 18:58:59.200+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-13 18:58:59.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.upload"
04-13 18:58:59.680+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.other /opt/usr/apps/org.tizen.other/bin/other
04-13 18:58:59.680+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
04-13 18:58:59.690+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
04-13 18:59:00.205+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-13 18:59:00.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
04-13 18:59:00.440+0900 E/socket.io( 2496): 566: Connected.
04-13 18:59:00.445+0900 D/sio_packet( 2496): from json
04-13 18:59:00.445+0900 D/value.IsObject()( 2496): if binary from json
04-13 18:59:00.445+0900 D/sio_packet( 2496): from json
04-13 18:59:00.445+0900 D/sio_packet( 2496): from json
04-13 18:59:00.445+0900 D/value.IsArray()( 2496): if arr from json
04-13 18:59:00.445+0900 D/sio_packet( 2496): from json
04-13 18:59:00.445+0900 D/sio_packet( 2496): IsInt64
04-13 18:59:00.445+0900 D/sio_packet( 2496): from json
04-13 18:59:00.445+0900 D/sio_packet( 2496): IsInt64
04-13 18:59:00.445+0900 E/socket.io( 2496): 554: On handshake, sid
04-13 18:59:00.460+0900 E/socket.io( 2496): 651: Received Message type(connect)
04-13 18:59:00.460+0900 E/socket.io( 2496): 489: On Connected
04-13 18:59:00.460+0900 F/sio_packet( 2496): accept()
04-13 18:59:00.460+0900 F/sio_packet( 2496): start accept_message
04-13 18:59:00.460+0900 F/sio_packet( 2496): arr, but we will use it like binary
04-13 18:59:00.460+0900 F/sio_packet( 2496): start accept_array_message()
04-13 18:59:00.460+0900 F/sio_packet( 2496): start accept_message
04-13 18:59:00.460+0900 F/sio_packet( 2496): start accept_message
04-13 18:59:00.460+0900 E/socket.io( 2496): 743: encoded paylod length: 13
04-13 18:59:00.465+0900 E/socket.io( 2496): 800: ping exit, con is expired? 0, ec: Operation canceled
04-13 18:59:00.625+0900 D/indicator( 2218): clock.c: indicator_get_apm_by_region(1016) > indicator_get_apm_by_region[1016]	 "TimeZone is Asia/Seoul"
04-13 18:59:00.625+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1136) > indicator_get_time_by_region[1136]	 "BestPattern is HH:mm"
04-13 18:59:00.625+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1137) > indicator_get_time_by_region[1137]	 "TimeZone is Asia/Seoul"
04-13 18:59:00.630+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1156) > indicator_get_time_by_region[1156]	 "DATE & TIME is en_US 18:59 5 HH:mm"
04-13 18:59:00.630+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1158) > indicator_get_time_by_region[1158]	 "24H :: Before change 18:59"
04-13 18:59:00.630+0900 D/indicator( 2218): clock.c: indicator_get_time_by_region(1160) > indicator_get_time_by_region[1160]	 "24H :: After change 18&#x2236;59"
04-13 18:59:00.630+0900 D/indicator( 2218): clock.c: indicator_clock_changed_cb(352) > indicator_clock_changed_cb[352]	 "[CLOCK MODULE] Timer Status : 366992 Time: <font_size=34>18&#x2236;59</font_size></font>"
04-13 18:59:01.205+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-13 18:59:01.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
04-13 18:59:01.570+0900 D/sio_packet( 2496): from json
04-13 18:59:01.570+0900 D/value.IsArray()( 2496): if arr from json
04-13 18:59:01.570+0900 D/sio_packet( 2496): from json
04-13 18:59:01.570+0900 D/sio_packet( 2496): from json
04-13 18:59:01.570+0900 D/value.IsObject()( 2496): if binary from json
04-13 18:59:01.570+0900 D/sio_packet( 2496): from json
04-13 18:59:01.570+0900 D/value.IsObject()( 2496): if binary from json
04-13 18:59:01.570+0900 D/sio_packet( 2496): from json
04-13 18:59:01.570+0900 D/sio_packet( 2496): IsInt64
04-13 18:59:01.570+0900 D/sio_packet( 2496): from json
04-13 18:59:01.570+0900 D/value.IsObject()( 2496): if binary from json
04-13 18:59:01.570+0900 F/value.IsObject()( 2496): num!!!!!0
04-13 18:59:01.570+0900 F/value.IsObject()( 2496): buf size!!!!!1
04-13 18:59:01.570+0900 E/socket.io( 2496): 669: Received Message type(Event)
04-13 18:59:01.570+0900 F/get_binary( 2496): in get binary_message()...
04-13 18:59:01.665+0900 D/CAPI_MEDIA_IMAGE_UTIL( 2496): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-13 18:59:01.665+0900 D/CAPI_MEDIA_IMAGE_UTIL( 2496): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] ERROR_NONE(0x00000000)
04-13 18:59:03.890+0900 W/CRASH_MANAGER( 2508): worker.c: worker_job(1189) > 11024966f7468142891914
