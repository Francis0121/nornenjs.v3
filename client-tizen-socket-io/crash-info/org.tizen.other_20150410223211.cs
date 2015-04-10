S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 15381
Date: 2015-04-10 22:32:11+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0x4

Register Information
r0   = 0x00000000, r1   = 0x00000000
r2   = 0xcd530c00, r3   = 0x00000000
r4   = 0x00000000, r5   = 0xb28e2154
r6   = 0xb28e2150, r7   = 0xb28e1ff8
r8   = 0xb70cc1a8, r9   = 0xb70ec904
r10  = 0xb6ef313c, fp   = 0xb28e2d9c
ip   = 0xb6f19b18, sp   = 0xb28e1ff8
lr   = 0xb3e7fc5b, pc   = 0xb3e02d3e
cpsr = 0x60000030

Memory Information
MemTotal:   797840 KB
MemFree:    441704 KB
Buffers:     30056 KB
Cached:     136320 KB
VmPeak:     100528 KB
VmSize:      93188 KB
VmLck:           0 KB
VmHWM:       21508 KB
VmRSS:       21508 KB
VmData:      38036 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24368 KB
VmPTE:          58 KB
VmSwap:          0 KB

Threads Information
Threads: 5
PID = 15381 TID = 15387
15381 15385 15386 15387 15388 

Maps Information
b3119000 b311a000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b3122000 b3129000 r-xp /usr/lib/libfeedback.so.0.1.4
b313c000 b313d000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b3145000 b315c000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b3297000 b329c000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b32a5000 b32af000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b32b8000 b32c3000 r-xp /usr/lib/evas/modules/engines/software_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3acc000 b3ad2000 r-xp /usr/lib/libUMP.so
b3ada000 b3ae1000 r-xp /usr/lib/libtbm.so.1.0.0
b3ae9000 b3aee000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3af6000 b3afd000 r-xp /usr/lib/libdrm.so.2.4.0
b3b06000 b3b08000 r-xp /usr/lib/libdri2.so.0.0.0
b3b10000 b3b23000 r-xp /usr/lib/libEGL_platform.so
b3b2c000 b3c03000 r-xp /usr/lib/libMali.so
b3c0e000 b3c15000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3c1d000 b3c22000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3c2a000 b3c41000 r-xp /usr/lib/libEGL.so.1.4
b3c4a000 b3c4f000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c58000 b3c59000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c62000 b3c7a000 r-xp /usr/lib/libpng12.so.0.50.0
b3c82000 b3cc0000 r-xp /usr/lib/libGLESv2.so.2.0
b3cc8000 b3ccc000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3cd5000 b3cd8000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3ce1000 b3d97000 r-xp /usr/lib/libcairo.so.2.11200.14
b3da2000 b3db8000 r-xp /usr/lib/libtts.so
b3dc1000 b3dd2000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3dda000 b3ddc000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3de4000 b3dec000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3df4000 b3eb1000 r-xp /opt/usr/apps/org.tizen.other/bin/other
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
b6f9c000 b7159000 rw-p [heap]
bec46000 bec67000 rwxp [stack]
End of Maps Information

Callstack Information (PID:15381)
Call Stack Count: 69
 0: sio::message::get_flag() const + 0x9 (0xb3e02d3e) [/opt/usr/apps/org.tizen.other/bin/other] + 0xed3e
 1: socket_io_client::{lambda(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)#2}::operator()(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&) const + 0x46 (0xb3e7fc5b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x8bc5b
 2: _ZNSt17_Function_handlerIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS5_EZ16socket_io_clientEUlS1_S7_bS8_E0_E9_M_invokeERKSt9_Any_da + 0x4e (0xb3e80747) [/opt/usr/apps/org.tizen.other/bin/other] + 0x8c747
 3: std::function<void (std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)>::operator()(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&) const + 0x5c (0xb3e1912d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2512d
 4: sio::event_adapter::adapt_func(std::function<void (std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)> const&, sio::event&) + 0x3a (0xb3e14053) [/opt/usr/apps/org.tizen.other/bin/other] + 0x20053
 5: _ZNSt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS6_EERNS4_5eventEESB_St12_PlaceholderILi1EEEE6__callIvISF_ + 0x54 (0xb3e2ebe9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3abe9
 6: _ZNSt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS6_EERNS4_5eventEESB_St12_PlaceholderILi1EEEEclIISF_EvEET0 + 0x2c (0xb3e26a8d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x32a8d
 7: _ZNSt17_Function_handlerIFvRN3sio5eventEESt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrINS0_7messageEEbRSA_EES2_ESF_St12_Plac + 0x24 (0xb3e1f851) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2b851
 8: std::function<void (sio::event&)>::operator()(sio::event&) const + 0x30 (0xb3e1b575) [/opt/usr/apps/org.tizen.other/bin/other] + 0x27575
 9: sio::client::impl::on_socketio_event(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int, std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&) + 0x9c (0xb3e01d01) [/opt/usr/apps/org.tizen.other/bin/other] + 0xdd01
10: sio::client::impl::on_decode(sio::packet const&) + 0x290 (0xb3e001b1) [/opt/usr/apps/org.tizen.other/bin/other] + 0xc1b1
11: std::_Mem_fn<void (sio::client::impl::*)(sio::packet const&)>::operator()(sio::client::impl*, sio::packet const&) const + 0x52 (0xb3e36a07) [/opt/usr/apps/org.tizen.other/bin/other] + 0x42a07
12: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvRKNS1_6packetEEEPS3_St12_PlaceholderILi1EEEE6__callIvIS6_EILi0ELi1EEEET_OSt5tupleII + 0x56 (0xb3e2f823) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3b823
13: void std::_Bind<std::_Mem_fn<void (sio::client::impl::*)(sio::packet const&)> (sio::client::impl*, std::_Placeholder<1>)>::operator()<sio::packet const&, void>(sio::packet const&) + 0x2c (0xb3e27c69) [/opt/usr/apps/org.tizen.other/bin/other] + 0x33c69
14: _ZNSt17_Function_handlerIFvRKN3sio6packetEESt5_BindIFSt7_Mem_fnIMNS0_6client4implEFvS3_EEPS8_St12_PlaceholderILi1EEEEE9_M_invoke + 0x24 (0xb3e20685) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c685
15: std::function<void (sio::packet const&)>::operator()(sio::packet const&) const + 0x30 (0xb3e77d89) [/opt/usr/apps/org.tizen.other/bin/other] + 0x83d89
16: sio::packet_manager::put_payload(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x126 (0xb3e73873) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7f873
17: _ZN3sio6client4impl10on_messageESt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS6_5alloc15con_msg_managerE + 0xda (0xb3dffe47) [/opt/usr/apps/org.tizen.other/bin/other] + 0xbe47
18: _ZNKSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS7_5alloc15con_msg_mana + 0x7e (0xb3e36937) [/opt/usr/apps/org.tizen.other/bin/other] + 0x42937
19: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS8_5alloc15con + 0x92 (0xb3e2f6bb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3b6bb
20: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS8_5alloc15con + 0x38 (0xb3e27a1d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x33a1d
21: _ZNSt17_Function_handlerIFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS4_5alloc15con_msg_managerEEEEES + 0x30 (0xb3e20441) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c441
22: std::function<void (std::weak_ptr<void>, std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager> >)>::operator()(std::weak_ptr<void>, std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager> >) const + 0x5c (0xb3e543e9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x603e9
23: websocketpp::connection<websocketpp::config::debug_asio>::handle_read_frame(std::error_code const&, unsigned int) + 0x744 (0xb3e4f9ed) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5b9ed
24: std::_Mem_fn<void (websocketpp::connection<websocketpp::config::debug_asio>::*)(std::error_code const&, unsigned int)>::operator()(websocketpp::connection<websocketpp::config::debug_asio>*, std::error_code const&, unsigned int) const + 0x62 (0xb3e64cbf) [/opt/usr/apps/org.tizen.other/bin/other] + 0x70cbf
25: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS1_6config10debug_asioEEEFvRKSt10error_codejEEPS5_St12_PlaceholderILi1EESD_ + 0x78 (0xb3e634b1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6f4b1
26: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS1_6config10debug_asioEEEFvRKSt10error_codejEEPS5_St12_PlaceholderILi1EESD_ + 0x38 (0xb3e60d09) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6cd09
27: _ZNSt17_Function_handlerIFvRKSt10error_codejESt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS6_6config10debug_asioEEEFvS2_jEE + 0x34 (0xb3e5d881) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69881
28: std::function<void (std::error_code const&, unsigned int)>::operator()(std::error_code const&, unsigned int) const + 0x42 (0xb3e44cd3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x50cd3
29: _ZN11websocketpp9transport4asio10connectionINS_6config10debug_asio16transport_configEE17handle_async_readERKN5boost6system10erro + 0x134 (0xb3e3eb71) [/opt/usr/apps/org.tizen.other/bin/other] + 0x4ab71
30: _ZNKSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS0_6config10debug_asio16transport_configEEEFvRKN5boost6system10error_ + 0x68 (0xb3e5e0cd) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6a0cd
31: _ZNKSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS0_6config10debug_asio16transport_configEEEFvRKN5boost6system10error_ + 0x30 (0xb3e5a72d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6672d
32: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS1_6config10debug_asio16transport_configEEEFvRKN5boost6syste + 0x76 (0xb3e6656b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7256b
33: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS1_6config10debug_asio16transport_configEEEFvRKN5boost6syste + 0x38 (0xb3e66035) [/opt/usr/apps/org.tizen.other/bin/other] + 0x72035
34: _ZN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS5_6config10debug_asio16transport + 0x20 (0xb3e65a61) [/opt/usr/apps/org.tizen.other/bin/other] + 0x71a61
35: _ZN5boost4asio19asio_handler_invokeINS0_6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6con + 0xc (0xb3e64e95) [/opt/usr/apps/org.tizen.other/bin/other] + 0x70e95
36: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10c + 0x1a (0xb3e63843) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6f843
37: _ZN5boost4asio6detail19asio_handler_invokeINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6con + 0x14 (0xb3e61429) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6d429
38: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10c + 0x1a (0xb3e5decf) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69ecf
39: _ZN5boost4asio6detail18completion_handlerINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6conf + 0x78 (0xb3e5dffd) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69ffd
40: _ZN5boost4asio6detail14strand_service8dispatchINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS7_ + 0x108 (0xb3e5a62d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6662d
41: _ZN5boost4asio10io_service6strand8dispatchINS0_6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionIN + 0x2e (0xb3e56a2b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x62a2b
42: _ZN5boost4asio6detail15wrapped_handlerINS0_10io_service6strandESt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionIN + 0x40 (0xb3e517e5) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5d7e5
43: _ZNSt17_Function_handlerIFvRKN5boost6system10error_codeEjENS0_4asio6detail15wrapped_handlerINS6_10io_service6strandESt5_BindIFSt + 0x34 (0xb3e4b92d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5792d
44: std::function<void (boost::system::error_code const&, unsigned int)>::operator()(boost::system::error_code const&, unsigned int) const + 0x42 (0xb3e5117b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5d17b
45: void websocketpp::transport::asio::custom_alloc_handler<std::function<void (boost::system::error_code const&, unsigned int)> >::operator()<boost::system::error_code, unsigned int>(boost::system::error_code, unsigned int) + 0x26 (0xb3e4b287) [/opt/usr/apps/org.tizen.other/bin/other] + 0x57287
46: _ZN5boost4asio6detail7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_serviceIS5_EEEENS0_17mutable_buffers_1EN + 0xee (0xb3e4ddbf) [/opt/usr/apps/org.tizen.other/bin/other] + 0x59dbf
47: _ZN5boost4asio6detail7binder2INS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_serviceIS6_EEEENS0_17mutabl + 0x26 (0xb3e66a7f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x72a7f
48: _ZN5boost4asio19asio_handler_invokeINS0_6detail7binder2INS2_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_s + 0xc (0xb3e664c9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x724c9
49: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2INS3_7read_opINS2_19basic_stream_socketINS2_2ip3tcpENS2 + 0x1a (0xb3e65fa7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x71fa7
50: _ZN5boost4asio6detail19asio_handler_invokeINS1_7binder2INS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_s + 0x18 (0xb3e65825) [/opt/usr/apps/org.tizen.other/bin/other] + 0x71825
51: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2INS3_7read_opINS2_19basic_stream_socketINS2_2ip3tcpENS2 + 0x1a (0xb3e646db) [/opt/usr/apps/org.tizen.other/bin/other] + 0x706db
52: _ZN5boost4asio6detail23reactive_socket_recv_opINS0_17mutable_buffers_1ENS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21 + 0x80 (0xb3e629f5) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6e9f5
53: boost::asio::detail::task_io_service_operation::complete(boost::asio::detail::task_io_service&, boost::system::error_code const&, unsigned int) + 0x1c (0xb3e0d319) [/opt/usr/apps/org.tizen.other/bin/other] + 0x19319
54: _ZN5boost4asio6detail13epoll_reactor16descriptor_state11do_completeEPNS1_15task_io_serviceEPNS1_25task_io_service_operationERKNS + 0x3c (0xb3e0ed69) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1ad69
55: boost::asio::detail::task_io_service_operation::complete(boost::asio::detail::task_io_service&, boost::system::error_code const&, unsigned int) + 0x1c (0xb3e0d319) [/opt/usr/apps/org.tizen.other/bin/other] + 0x19319
56: _ZN5boost4asio6detail15task_io_service10do_run_oneERNS1_11scoped_lockINS1_11posix_mutexEEERNS1_27task_io_service_thread_infoERKN + 0x122 (0xb3e0f67f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1b67f
57: boost::asio::detail::task_io_service::run(boost::system::error_code&) + 0xd8 (0xb3e0f219) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1b219
58: boost::asio::io_service::run() + 0x22 (0xb3e0f92f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1b92f
59: websocketpp::transport::asio::endpoint<websocketpp::config::debug_asio::transport_config>::run() + 0x12 (0xb3e1b403) [/opt/usr/apps/org.tizen.other/bin/other] + 0x27403
60: sio::client::impl::run_loop() + 0x14 (0xb3e01c2d) [/opt/usr/apps/org.tizen.other/bin/other] + 0xdc2d
61: std::_Mem_fn<void (sio::client::impl::*)()>::operator()(sio::client::impl*) const + 0x44 (0xb3e71b8d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7db8d
62: void std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)>::__call<void, , 0>(std::tuple<>&&, std::_Index_tuple<0>) + 0x36 (0xb3e717d7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7d7d7
63: void std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)>::operator()<, void>() + 0x1a (0xb3e71497) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7d497
64: _ZNSt12_Bind_resultIvFSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS4_EEvEE6__callIvIEIEEEvOSt5tupleIIDpT0_EESt12_Index_tupleII + 0x16 (0xb3e70f0b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7cf0b
65: void std::_Bind_result<void, std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)> ()>::operator()<>() + 0x1e (0xb3e7034b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7c34b
66: std::thread::_Impl<std::_Bind_result<void, std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)> ()> >::_M_run() + 0x14 (0xb3e6e051) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7a051
67: (0xb5b8c7dd) [/usr/lib/libstdc++.so.6] + 0x517dd
68: (0xb6ee7d30) [/lib/libpthread.so.0] + 0x6d30
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
t : 0"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:06.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1211) > handle_more_notify_icon[1211]	 "LANDSCAPE :: 2"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1220) > handle_more_notify_icon[1220]	 "LANDSCAPE :: handle_more_notify_hide"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.show"
04-10 22:32:06.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.wifi.show"
04-10 22:32:07.195+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-10 22:32:07.195+0900 D/indicator( 2218): wifi.c: show_wifi_transfer_icon(169) > show_wifi_transfer_icon[169]	 "same transfer state"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:07.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1197) > handle_more_notify_icon[1197]	 "PORT :: 2"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1206) > handle_more_notify_icon[1206]	 "PORT :: handle_more_notify_hide"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:07.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1211) > handle_more_notify_icon[1211]	 "LANDSCAPE :: 2"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1220) > handle_more_notify_icon[1220]	 "LANDSCAPE :: handle_more_notify_hide"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.show"
04-10 22:32:07.205+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.wifi.show"
04-10 22:32:08.195+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-10 22:32:08.195+0900 D/indicator( 2218): wifi.c: show_wifi_transfer_icon(169) > show_wifi_transfer_icon[169]	 "same transfer state"
04-10 22:32:08.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:08.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:08.200+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
04-10 22:32:08.205+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1197) > handle_more_notify_icon[1197]	 "PORT :: 2"
04-10 22:32:08.210+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1206) > handle_more_notify_icon[1206]	 "PORT :: handle_more_notify_hide"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:08.215+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1211) > handle_more_notify_icon[1211]	 "LANDSCAPE :: 2"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1220) > handle_more_notify_icon[1220]	 "LANDSCAPE :: handle_more_notify_hide"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.show"
04-10 22:32:08.220+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.wifi.show"
04-10 22:32:10.145+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 0
04-10 22:32:10.150+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-10 22:32:10.185+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 15378, pid = 15380
04-10 22:32:10.200+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-10 22:32:10.200+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.download"
04-10 22:32:10.220+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-10 22:32:10.245+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
04-10 22:32:10.245+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-10 22:32:10.245+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-10 22:32:10.245+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-10 22:32:10.245+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-10 22:32:10.245+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-10 22:32:10.245+0900 D/AUL_PAD ( 2204): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-10 22:32:10.255+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 15381 /opt/usr/apps/org.tizen.other/bin/other
04-10 22:32:10.255+0900 D/AUL_PAD (15381): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-10 22:32:10.255+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-10 22:32:10.255+0900 D/AUL_PAD (15381): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-10 22:32:10.260+0900 D/AUL_PAD (15381): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-10 22:32:10.355+0900 D/AUL_PAD (15381): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-10 22:32:10.355+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-10 22:32:10.355+0900 D/AUL_PAD (15381): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-10 22:32:10.355+0900 D/AUL_PAD (15381): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-10 22:32:10.355+0900 D/AUL_PAD (15381): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-10 22:32:10.355+0900 D/LAUNCH  (15381): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-10 22:32:10.400+0900 I/CAPI_APPFW_APPLICATION(15381): app_main.c: ui_app_main(697) > app_efl_main
04-10 22:32:10.400+0900 D/LAUNCH  (15381): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-10 22:32:10.435+0900 D/APP_CORE(15381): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-10 22:32:10.440+0900 D/AUL     (15381): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 15381 is org.tizen.other
04-10 22:32:10.440+0900 D/APP_CORE(15381): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-10 22:32:10.445+0900 D/APP_CORE(15381): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-10 22:32:10.445+0900 D/AUL     (15381): app_sock.c: __create_server_sock(135) > pg path - already exists
04-10 22:32:10.445+0900 D/LAUNCH  (15381): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-10 22:32:10.445+0900 I/CAPI_APPFW_APPLICATION(15381): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-10 22:32:10.455+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-10 22:32:10.455+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-10 22:32:10.455+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-10 22:32:10.455+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-10 22:32:10.455+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-10 22:32:10.455+0900 D/RESOURCED( 2372): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-10 22:32:10.455+0900 D/RESOURCED( 2372): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
04-10 22:32:10.455+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 15381, type 4 
04-10 22:32:10.455+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 15381
04-10 22:32:10.455+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-10 22:32:10.455+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 15381
04-10 22:32:10.455+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-10 22:32:10.455+0900 D/RESOURCED( 2372): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 15381
04-10 22:32:10.455+0900 D/RESOURCED( 2372): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 15381
04-10 22:32:10.455+0900 D/RESOURCED( 2372): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-10 22:32:10.455+0900 D/RESOURCED( 2372): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-10 22:32:10.455+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-10 22:32:10.455+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-10 22:32:10.455+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-10 22:32:10.455+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-10 22:32:10.455+0900 E/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 574
04-10 22:32:10.455+0900 D/RESOURCED( 2372): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-10 22:32:10.545+0900 F/socket.c(15381): Rapidson Test::{"project":"rapidjson","stars":11}
04-10 22:32:10.545+0900 F/socket.io(15381): thread_start
04-10 22:32:10.550+0900 D/LAUNCH  (15381): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-10 22:32:10.550+0900 D/APP_CORE(15381): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-10 22:32:10.550+0900 D/APP_CORE(15381): appcore.c: __aul_handler(423) > [APP 15381]     AUL event: AUL_START
04-10 22:32:10.550+0900 D/APP_CORE(15381): appcore-efl.c: __do_app(470) > [APP 15381] Event: RESET State: CREATED
04-10 22:32:10.550+0900 D/APP_CORE(15381): appcore-efl.c: __do_app(496) > [APP 15381] RESET
04-10 22:32:10.550+0900 D/LAUNCH  (15381): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-10 22:32:10.550+0900 I/CAPI_APPFW_APPLICATION(15381): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-10 22:32:10.550+0900 D/APP_SVC (15381): appsvc.c: __set_bundle(161) > __set_bundle
04-10 22:32:10.550+0900 F/socket.io(15381): thread_start
04-10 22:32:10.550+0900 F/socket.io(15381): finish 0
04-10 22:32:10.550+0900 D/LAUNCH  (15381): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-10 22:32:10.550+0900 F/socket.io(15381): Connect Start
04-10 22:32:10.550+0900 F/socket.io(15381): Set ConnectListener
04-10 22:32:10.550+0900 F/socket.io(15381): Set ClosetListener
04-10 22:32:10.550+0900 F/socket.io(15381): Set FaileListener
04-10 22:32:10.550+0900 F/socket.io(15381): Connect
04-10 22:32:10.550+0900 F/socket.io(15381): Lock
04-10 22:32:10.550+0900 F/socket.io(15381): !!!
04-10 22:32:10.565+0900 I/APP_CORE(15381): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-10 22:32:10.565+0900 I/APP_CORE(15381): appcore-efl.c: __do_app(509) > [APP 15381] Initial Launching, call the resume_cb
04-10 22:32:10.565+0900 I/CAPI_APPFW_APPLICATION(15381): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-10 22:32:10.565+0900 D/APP_CORE(15381): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-10 22:32:10.590+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=15381
04-10 22:32:10.625+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-10 22:32:10.630+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600002"
04-10 22:32:10.630+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-10 22:32:10.630+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600002"
04-10 22:32:10.630+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-10 22:32:10.630+0900 D/APP_CORE(15381): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600002
04-10 22:32:10.630+0900 D/APP_CORE(15381): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600002
04-10 22:32:10.745+0900 D/APP_CORE(15381): appcore.c: __prt_ltime(183) > [APP 15381] first idle after reset: 605 msec
04-10 22:32:10.765+0900 E/socket.io(15381): 566: Connected.
04-10 22:32:10.765+0900 E/socket.io(15381): 554: On handshake, sid
04-10 22:32:10.765+0900 E/socket.io(15381): 651: Received Message type(connect)
04-10 22:32:10.765+0900 E/socket.io(15381): 489: On Connected
04-10 22:32:10.765+0900 F/socket.io(15381): unlock
04-10 22:32:10.765+0900 F/socket.io(15381): emit connectMessage
04-10 22:32:10.765+0900 E/socket.io(15381): 743: encoded paylod length: 63
04-10 22:32:10.765+0900 F/socket.io(15381): bind connectMessage
04-10 22:32:10.765+0900 F/socket.io(15381): 1
04-10 22:32:10.765+0900 F/socket.io(15381): 2
04-10 22:32:10.765+0900 F/socket.io(15381): 3
04-10 22:32:10.765+0900 F/socket.io(15381): emit connectMessage
04-10 22:32:10.765+0900 E/socket.io(15381): 743: encoded paylod length: 13
04-10 22:32:10.765+0900 F/socket.io(15381): 4
04-10 22:32:10.765+0900 F/socket.io(15381): 5
04-10 22:32:10.765+0900 F/socket.io(15381): 6
04-10 22:32:10.765+0900 F/socket.io(15381): close 15381
04-10 22:32:10.765+0900 E/socket.io(15381): 800: ping exit, con is expired? 0, ec: Operation canceled
04-10 22:32:10.765+0900 E/socket.io(15381): 800: ping exit, con is expired? 0, ec: Operation canceled
04-10 22:32:10.775+0900 E/socket.io(15381): 669: Received Message type(Event)
04-10 22:32:10.775+0900 F/socket.io(15381): bind_event [connectMessage] 15381
04-10 22:32:10.775+0900 F/socket.io(15381): connectMessage :: 
04-10 22:32:10.790+0900 D/APP_CORE( 2245): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 1
04-10 22:32:10.790+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
04-10 22:32:10.790+0900 D/APP_CORE( 2245): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
04-10 22:32:10.790+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(470) > [APP 2245] Event: PAUSE State: RUNNING
04-10 22:32:10.790+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(538) > [APP 2245] PAUSE
04-10 22:32:10.790+0900 I/CAPI_APPFW_APPLICATION( 2245): app_main.c: app_appcore_pause(195) > app_appcore_pause
04-10 22:32:10.790+0900 D/MENU_SCREEN( 2245): menu_screen.c: _pause_cb(538) > Pause start
04-10 22:32:10.790+0900 D/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-10 22:32:10.790+0900 E/APP_CORE( 2245): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-10 22:32:10.790+0900 D/APP_CORE(15381): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600002 fully_obscured 0
04-10 22:32:10.790+0900 D/APP_CORE(15381): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-10 22:32:10.790+0900 D/APP_CORE(15381): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-10 22:32:10.790+0900 D/APP_CORE(15381): appcore-efl.c: __do_app(470) > [APP 15381] Event: RESUME State: RUNNING
04-10 22:32:10.790+0900 D/LAUNCH  (15381): appcore-efl.c: __do_app(557) > [other:Application:resume:start]
04-10 22:32:10.790+0900 D/LAUNCH  (15381): appcore-efl.c: __do_app(567) > [other:Application:resume:done]
04-10 22:32:10.790+0900 D/LAUNCH  (15381): appcore-efl.c: __do_app(569) > [other:Application:Launching:done]
04-10 22:32:10.790+0900 D/APP_CORE(15381): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-10 22:32:10.790+0900 E/APP_CORE(15381): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-10 22:32:10.790+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2245, type = 2
04-10 22:32:10.795+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 15381, type = 0
04-10 22:32:10.795+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 15381
04-10 22:32:10.795+0900 I/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 15381, oom : 200
04-10 22:32:10.795+0900 E/RESOURCED( 2372): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
04-10 22:32:10.795+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(15381) status(3)
04-10 22:32:11.185+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-10 22:32:11.185+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.updownload"
04-10 22:32:11.465+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.other /opt/usr/apps/org.tizen.other/bin/other
04-10 22:32:11.465+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
04-10 22:32:11.475+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
04-10 22:32:11.870+0900 E/socket.io(15381): 669: Received Message type(Event)
04-10 22:32:11.870+0900 F/socket.io(15381): bind_event [test1] 15381
04-10 22:32:12.185+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
04-10 22:32:12.185+0900 D/indicator( 2218): wifi.c: show_wifi_transfer_icon(169) > show_wifi_transfer_icon[169]	 "same transfer state"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1297) > indicator_get_noti_list_index_count[1297]	 "System Count : 0, Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_noti_list_index_count(1392) > indicator_get_noti_list_index_count[1392]	 "Notification icon ret 5"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1197) > handle_more_notify_icon[1197]	 "PORT :: 2"
04-10 22:32:12.190+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1206) > handle_more_notify_icon[1206]	 "PORT :: handle_more_notify_hide"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1399) > indicator_get_system_list_index_count[1399]	 "indicator_get_system_list_index_count"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_system_list_index_count(1417) > indicator_get_system_list_index_count[1417]	 "Noti count : 2 , MiniCtrl count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1453) > indicator_get_minictrl_list_index_count[1453]	 "indicator_get_minictrl_list_index_count"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_minictrl_list_index_count(1486) > indicator_get_minictrl_list_index_count[1486]	 "Noti count : 2 , System count : 0 , Minictrl ret : 3"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.system.hide"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.system.hide"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.minictrl.hide"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.minictrl.hide"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.noti.show"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.noti.show"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1177) > handle_more_notify_icon[1177]	 "handle_more_notify_icon called !!"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_box_util.c: indicator_get_count_in_minictrl_list(1146) > indicator_get_count_in_minictrl_list[1146]	 "Minictrl Count : 0"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1194) > handle_more_notify_icon[1194]	 "System count : 0, Minictrl count : 0, Notification count : 2"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1211) > handle_more_notify_icon[1211]	 "LANDSCAPE :: 2"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_icon_util.c: handle_more_notify_icon(1220) > handle_more_notify_icon[1220]	 "LANDSCAPE :: handle_more_notify_hide"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.show"
04-10 22:32:12.195+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(107) > [SECURE_LOG] indicator_signal_emit[107]	 "emission indicator.wifi.show"
04-10 22:32:15.860+0900 D/APP_CORE( 2245): appcore-efl.c: __do_app(470) > [APP 2245] Event: MEM_FLUSH State: PAUSED
04-10 22:32:17.325+0900 W/CRASH_MANAGER(15390): worker.c: worker_job(1189) > 11153816f7468142867273
