S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 3913
Date: 2015-04-13 20:27:19+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = (nil)

Register Information
r0   = 0x00000000, r1   = 0x00002460
r2   = 0xb71b3540, r3   = 0x00000000
r4   = 0xb3f401d4, r5   = 0xb0e01078
r6   = 0xb0dfe148, r7   = 0xb0dfdf70
r8   = 0xb0e00af8, r9   = 0xb0e00ab4
r10  = 0xb6f7913c, fp   = 0xb0dfed9c
ip   = 0xb3f40498, sp   = 0xb0dfdf50
lr   = 0xb3eff5ef, pc   = 0xb3e85c1e
cpsr = 0x60000030

Memory Information
MemTotal:   797840 KB
MemFree:    470652 KB
Buffers:     45536 KB
Cached:      84632 KB
VmPeak:     149332 KB
VmSize:     123884 KB
VmLck:           0 KB
VmHWM:       19516 KB
VmRSS:       19516 KB
VmData:      67996 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24984 KB
VmPTE:          90 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 3913 TID = 3922
3913 3917 3918 3919 3920 3921 3922 3923 

Maps Information
b1830000 b1831000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b186e000 b186f000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b1877000 b187e000 r-xp /usr/lib/libfeedback.so.0.1.4
b1891000 b1892000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b189a000 b18b1000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b1a37000 b1a3c000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b3285000 b32d0000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b32d9000 b32e3000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b32ec000 b3348000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3b55000 b3b5b000 r-xp /usr/lib/libUMP.so
b3b63000 b3b76000 r-xp /usr/lib/libEGL_platform.so
b3b7f000 b3c56000 r-xp /usr/lib/libMali.so
b3c61000 b3c78000 r-xp /usr/lib/libEGL.so.1.4
b3c81000 b3c86000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c8f000 b3c90000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c99000 b3cb1000 r-xp /usr/lib/libpng12.so.0.50.0
b3cb9000 b3cf7000 r-xp /usr/lib/libGLESv2.so.2.0
b3cff000 b3d03000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3d0c000 b3d0e000 r-xp /usr/lib/libdri2.so.0.0.0
b3d16000 b3d1d000 r-xp /usr/lib/libdrm.so.2.4.0
b3d26000 b3ddc000 r-xp /usr/lib/libcairo.so.2.11200.14
b3de7000 b3dfd000 r-xp /usr/lib/libtts.so
b3e06000 b3e0d000 r-xp /usr/lib/libtbm.so.1.0.0
b3e15000 b3e1a000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3e22000 b3e33000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3e3b000 b3e42000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3e4a000 b3e4f000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3e57000 b3e59000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3e61000 b3e69000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3e71000 b3e74000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3e7d000 b3f37000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3f41000 b3f4b000 r-xp /lib/libnss_files-2.13.so
b3f59000 b3f5b000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4164000 b4185000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b418e000 b41ab000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b41b4000 b4282000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4299000 b42bf000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b42c9000 b42cb000 r-xp /usr/lib/libiniparser.so.0
b42d5000 b42db000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b42e4000 b42ea000 r-xp /usr/lib/libappsvc.so.0.1.0
b42f3000 b42f5000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b42fe000 b4302000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b430a000 b430e000 r-xp /usr/lib/libogg.so.0.7.1
b4316000 b4338000 r-xp /usr/lib/libvorbis.so.0.4.3
b4340000 b4424000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b4438000 b4469000 r-xp /usr/lib/libFLAC.so.8.2.0
b4472000 b4474000 r-xp /usr/lib/libXau.so.6.0.0
b447c000 b44c8000 r-xp /usr/lib/libssl.so.1.0.0
b44d5000 b4503000 r-xp /usr/lib/libidn.so.11.5.44
b450b000 b4515000 r-xp /usr/lib/libcares.so.2.1.0
b451d000 b4562000 r-xp /usr/lib/libsndfile.so.1.0.25
b4570000 b4577000 r-xp /usr/lib/libsensord-share.so
b457f000 b4595000 r-xp /lib/libexpat.so.1.5.2
b45a3000 b45a6000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b45ae000 b45e2000 r-xp /usr/lib/libicule.so.51.1
b45eb000 b45fe000 r-xp /usr/lib/libxcb.so.1.1.0
b4606000 b4641000 r-xp /usr/lib/libcurl.so.4.3.0
b464a000 b4653000 r-xp /usr/lib/libethumb.so.1.7.99
b5bc1000 b5c55000 r-xp /usr/lib/libstdc++.so.6.0.16
b5c68000 b5c6a000 r-xp /usr/lib/libctxdata.so.0.0.0
b5c72000 b5c7f000 r-xp /usr/lib/libremix.so.0.0.0
b5c87000 b5c88000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c90000 b5ca7000 r-xp /usr/lib/liblua-5.1.so
b5cb0000 b5cb7000 r-xp /usr/lib/libembryo.so.1.7.99
b5cbf000 b5ce2000 r-xp /usr/lib/libjpeg.so.8.0.2
b5cfa000 b5d10000 r-xp /usr/lib/libsensor.so.1.1.0
b5d19000 b5d6f000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5d7c000 b5d9f000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5da8000 b5dee000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5df7000 b5e0a000 r-xp /usr/lib/libfribidi.so.0.3.1
b5e12000 b5e62000 r-xp /usr/lib/libfreetype.so.6.8.1
b5e6d000 b5e70000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5e78000 b5e7c000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e84000 b5e89000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e92000 b5e9c000 r-xp /usr/lib/libXext.so.6.4.0
b5ea4000 b5f85000 r-xp /usr/lib/libX11.so.6.3.0
b5f90000 b5f93000 r-xp /usr/lib/libXtst.so.6.1.0
b5f9b000 b5fa1000 r-xp /usr/lib/libXrender.so.1.3.0
b5fa9000 b5fae000 r-xp /usr/lib/libXrandr.so.2.2.0
b5fb6000 b5fb7000 r-xp /usr/lib/libXinerama.so.1.0.0
b5fc0000 b5fc8000 r-xp /usr/lib/libXi.so.6.1.0
b5fc9000 b5fcc000 r-xp /usr/lib/libXfixes.so.3.1.0
b5fd4000 b5fd6000 r-xp /usr/lib/libXgesture.so.7.0.0
b5fde000 b5fe0000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5fe8000 b5fe9000 r-xp /usr/lib/libXdamage.so.1.1.0
b5ff2000 b5ff8000 r-xp /usr/lib/libXcursor.so.1.0.2
b6001000 b601a000 r-xp /usr/lib/libecore_con.so.1.7.99
b6024000 b602a000 r-xp /usr/lib/libecore_imf.so.1.7.99
b6032000 b603a000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6042000 b6046000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b604f000 b6065000 r-xp /usr/lib/libefreet.so.1.7.99
b606e000 b6077000 r-xp /usr/lib/libedbus.so.1.7.99
b607f000 b6164000 r-xp /usr/lib/libicuuc.so.51.1
b6179000 b62b8000 r-xp /usr/lib/libicui18n.so.51.1
b62c8000 b6324000 r-xp /usr/lib/libedje.so.1.7.99
b632e000 b633f000 r-xp /usr/lib/libecore_input.so.1.7.99
b6347000 b634c000 r-xp /usr/lib/libecore_file.so.1.7.99
b6354000 b636d000 r-xp /usr/lib/libeet.so.1.7.99
b637e000 b6382000 r-xp /usr/lib/libappcore-common.so.1.1
b638b000 b6457000 r-xp /usr/lib/libevas.so.1.7.99
b647c000 b649d000 r-xp /usr/lib/libecore_evas.so.1.7.99
b64a6000 b64d5000 r-xp /usr/lib/libecore_x.so.1.7.99
b64df000 b6613000 r-xp /usr/lib/libelementary.so.1.7.99
b662b000 b662c000 r-xp /usr/lib/libjournal.so.0.1.0
b6635000 b6700000 r-xp /usr/lib/libxml2.so.2.7.8
b670e000 b671e000 r-xp /lib/libresolv-2.13.so
b6722000 b6738000 r-xp /lib/libz.so.1.2.5
b6740000 b6742000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b674a000 b674f000 r-xp /usr/lib/libffi.so.5.0.10
b6758000 b6759000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6761000 b6764000 r-xp /lib/libattr.so.1.1.0
b676c000 b6914000 r-xp /usr/lib/libcrypto.so.1.0.0
b6934000 b694e000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b6957000 b69c0000 r-xp /lib/libm-2.13.so
b69c9000 b6a09000 r-xp /usr/lib/libeina.so.1.7.99
b6a12000 b6a1a000 r-xp /usr/lib/libvconf.so.0.2.45
b6a22000 b6a25000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6a2d000 b6a61000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a6a000 b6b3e000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b4a000 b6b50000 r-xp /lib/librt-2.13.so
b6b59000 b6b5e000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6b67000 b6b6e000 r-xp /lib/libcrypt-2.13.so
b6b9e000 b6ba1000 r-xp /lib/libcap.so.2.21
b6ba9000 b6bab000 r-xp /usr/lib/libiri.so
b6bb3000 b6bd2000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6bda000 b6bf0000 r-xp /usr/lib/libecore.so.1.7.99
b6c06000 b6c0b000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6c14000 b6ce4000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6ce5000 b6cf3000 r-xp /usr/lib/libail.so.0.1.0
b6cfb000 b6d12000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6d1b000 b6d25000 r-xp /lib/libunwind.so.8.0.1
b6d53000 b6e6e000 r-xp /lib/libc-2.13.so
b6e7c000 b6e84000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e8c000 b6eb6000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6ebf000 b6ec2000 r-xp /usr/lib/libbundle.so.0.1.22
b6eca000 b6ecc000 r-xp /lib/libdl-2.13.so
b6ed5000 b6ed8000 r-xp /usr/lib/libsmack.so.1.0.0
b6ee0000 b6f42000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f4c000 b6f5e000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f67000 b6f7b000 r-xp /lib/libpthread-2.13.so
b6f88000 b6f8c000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f96000 b6f98000 r-xp /usr/lib/libdlog.so.0.0.0
b6fa0000 b6fab000 r-xp /usr/lib/libaul.so.0.1.0
b6fb5000 b6fb9000 r-xp /usr/lib/libsys-assert.so
b6fc2000 b6fdf000 r-xp /lib/ld-2.13.so
b6fe8000 b6fee000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6ff6000 b7022000 rw-p [heap]
b7022000 b72c1000 rw-p [heap]
bef09000 bef2a000 rwxp [stack]
End of Maps Information

Callstack Information (PID:3913)
Call Stack Count: 69
 0: setTextureData + 0xd (0xb3e85c1e) [/opt/usr/apps/org.tizen.other/bin/other] + 0x8c1e
 1: socket_io_client::{lambda(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)#1}::operator()(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&) const + 0x20a (0xb3eff5ef) [/opt/usr/apps/org.tizen.other/bin/other] + 0x825ef
 2: _ZNSt17_Function_handlerIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS5_EZ16socket_io_clientEUlS1_S7_bS8_E_E9_M_invokeERKSt9_Any_dat + 0x4a (0xb3effc2f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x82c2f
 3: std::function<void (std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)>::operator()(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&) const + 0x5a (0xb3ea0b8b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x23b8b
 4: sio::event_adapter::adapt_func(std::function<void (std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&, bool, std::shared_ptr<sio::message>&)> const&, sio::event&) + 0x3a (0xb3e9becb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1eecb
 5: _ZNSt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS6_EERNS4_5eventEESB_St12_PlaceholderILi1EEEE6__callIvISF_ + 0x4e (0xb3eb6cab) [/opt/usr/apps/org.tizen.other/bin/other] + 0x39cab
 6: _ZNSt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrIN3sio7messageEEbRS6_EERNS4_5eventEESB_St12_PlaceholderILi1EEEEclIISF_EvEET0 + 0x2c (0xb3eaf261) [/opt/usr/apps/org.tizen.other/bin/other] + 0x32261
 7: _ZNSt17_Function_handlerIFvRN3sio5eventEESt5_BindIFPFvRKSt8functionIFvRKSsRKSt10shared_ptrINS0_7messageEEbRSA_EES2_ESF_St12_Plac + 0x22 (0xb3ea74bb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2a4bb
 8: std::function<void (sio::event&)>::operator()(sio::event&) const + 0x30 (0xb3ea379d) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2679d
 9: sio::client::impl::on_socketio_event(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int, std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::shared_ptr<sio::message> const&) + 0x96 (0xb3e8b10b) [/opt/usr/apps/org.tizen.other/bin/other] + 0xe10b
10: sio::client::impl::on_decode(sio::packet const&) + 0x264 (0xb3e896b9) [/opt/usr/apps/org.tizen.other/bin/other] + 0xc6b9
11: void std::_Mem_fn<void (sio::client::impl::*)(sio::packet const&)>::operator()<sio::packet const&, void>(sio::client::impl*, sio::packet const&) const + 0x4a (0xb3ebe6f7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x416f7
12: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvRKNS1_6packetEEEPS3_St12_PlaceholderILi1EEEE6__callIvIS6_EILj0ELj1EEEET_OSt5tupleII + 0x52 (0xb3eb7867) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3a867
13: void std::_Bind<std::_Mem_fn<void (sio::client::impl::*)(sio::packet const&)> (sio::client::impl*, std::_Placeholder<1>)>::operator()<sio::packet const&, void>(sio::packet const&) + 0x2c (0xb3eb00c1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x330c1
14: _ZNSt17_Function_handlerIFvRKN3sio6packetEESt5_BindIFSt7_Mem_fnIMNS0_6client4implEFvS3_EEPS8_St12_PlaceholderILi1EEEEE9_M_invoke + 0x22 (0xb3ea83f3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2b3f3
15: std::function<void (sio::packet const&)>::operator()(sio::packet const&) const + 0x30 (0xb3ef7ebd) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7aebd
16: sio::packet_manager::put_payload(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x11a (0xb3ef418f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x7718f
17: _ZN3sio6client4impl10on_messageESt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS6_5alloc15con_msg_managerE + 0xd4 (0xb3e89385) [/opt/usr/apps/org.tizen.other/bin/other] + 0xc385
18: _ZNKSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS7_5alloc15con_msg_mana + 0x76 (0xb3ebe603) [/opt/usr/apps/org.tizen.other/bin/other] + 0x41603
19: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS8_5alloc15con + 0x6e (0xb3eb771b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x3a71b
20: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS8_5alloc15con + 0x38 (0xb3eafebd) [/opt/usr/apps/org.tizen.other/bin/other] + 0x32ebd
21: _ZNSt17_Function_handlerIFvSt8weak_ptrIvESt10shared_ptrIN11websocketpp14message_buffer7messageINS4_5alloc15con_msg_managerEEEEES + 0x2e (0xb3ea81ef) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2b1ef
22: std::function<void (std::weak_ptr<void>, std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager> >)>::operator()(std::weak_ptr<void>, std::shared_ptr<websocketpp::message_buffer::message<websocketpp::message_buffer::alloc::con_msg_manager> >) const + 0x5c (0xb3ed6981) [/opt/usr/apps/org.tizen.other/bin/other] + 0x59981
23: websocketpp::connection<websocketpp::config::debug_asio>::handle_read_frame(std::error_code const&, unsigned int) + 0x6d0 (0xb3ed2e59) [/opt/usr/apps/org.tizen.other/bin/other] + 0x55e59
24: void std::_Mem_fn<void (websocketpp::connection<websocketpp::config::debug_asio>::*)(std::error_code const&, unsigned int)>::operator()<std::error_code const&, unsigned int, void>(websocketpp::connection<websocketpp::config::debug_asio>*, std::error_code const&, unsigned int&&) const + 0x58 (0xb3ee8f71) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6bf71
25: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS1_6config10debug_asioEEEFvRKSt10error_codejEEPS5_St12_PlaceholderILi1EESD_ + 0x6e (0xb3ee82df) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6b2df
26: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS1_6config10debug_asioEEEFvRKSt10error_codejEEPS5_St12_PlaceholderILi1EESD_ + 0x38 (0xb3ee7329) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6a329
27: _ZNSt17_Function_handlerIFvRKSt10error_codejESt5_BindIFSt7_Mem_fnIMN11websocketpp10connectionINS6_6config10debug_asioEEEFvS2_jEE + 0x30 (0xb3ee6349) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69349
28: std::function<void (std::error_code const&, unsigned int)>::operator()(std::error_code const&, unsigned int) const + 0x40 (0xb3eca125) [/opt/usr/apps/org.tizen.other/bin/other] + 0x4d125
29: _ZN11websocketpp9transport4asio10connectionINS_6config10debug_asio16transport_configEE17handle_async_readERKN5boost6system10erro + 0x126 (0xb3ec5017) [/opt/usr/apps/org.tizen.other/bin/other] + 0x48017
30: _ZNKSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS0_6config10debug_asio16transport_configEEEFvRKN5boost6system10error_ + 0x5e (0xb3ee868f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6b68f
31: _ZNKSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS0_6config10debug_asio16transport_configEEEFvRKN5boost6system10error_ + 0x34 (0xb3ee7915) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6a915
32: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS1_6config10debug_asio16transport_configEEEFvRKN5boost6syste + 0x6c (0xb3ee67f5) [/opt/usr/apps/org.tizen.other/bin/other] + 0x697f5
33: _ZNSt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS1_6config10debug_asio16transport_configEEEFvRKN5boost6syste + 0x38 (0xb3ee5665) [/opt/usr/apps/org.tizen.other/bin/other] + 0x68665
34: _ZN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS5_6config10debug_asio16transport + 0x1e (0xb3ee427b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6727b
35: _ZN5boost4asio19asio_handler_invokeINS0_6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6con + 0xc (0xb3ee2be1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x65be1
36: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10c + 0x1a (0xb3ee185f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6485f
37: _ZN5boost4asio6detail19asio_handler_invokeINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6con + 0x14 (0xb3edfc39) [/opt/usr/apps/org.tizen.other/bin/other] + 0x62c39
38: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10c + 0x1a (0xb3edd593) [/opt/usr/apps/org.tizen.other/bin/other] + 0x60593
39: _ZN5boost4asio6detail18completion_handlerINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS6_6conf + 0x72 (0xb3edd69b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6069b
40: _ZN5boost4asio6detail14strand_service8dispatchINS1_7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionINS7_ + 0xf4 (0xb3edaf15) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5df15
41: _ZN5boost4asio10io_service6strand8dispatchINS0_6detail7binder2ISt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionIN + 0x2c (0xb3ed8161) [/opt/usr/apps/org.tizen.other/bin/other] + 0x5b161
42: _ZN5boost4asio6detail15wrapped_handlerINS0_10io_service6strandESt5_BindIFSt7_Mem_fnIMN11websocketpp9transport4asio10connectionIN + 0x3e (0xb3ed44c3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x574c3
43: _ZNSt17_Function_handlerIFvRKN5boost6system10error_codeEjENS0_4asio6detail15wrapped_handlerINS6_10io_service6strandESt5_BindIFSt + 0x30 (0xb3ecf975) [/opt/usr/apps/org.tizen.other/bin/other] + 0x52975
44: std::function<void (boost::system::error_code const&, unsigned int)>::operator()(boost::system::error_code const&, unsigned int) const + 0x40 (0xb3ed3fc1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x56fc1
45: void websocketpp::transport::asio::custom_alloc_handler<std::function<void (boost::system::error_code const&, unsigned int)> >::operator()<boost::system::error_code, unsigned int>(boost::system::error_code, unsigned int) + 0x20 (0xb3ecf515) [/opt/usr/apps/org.tizen.other/bin/other] + 0x52515
46: _ZN5boost4asio6detail7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_serviceIS5_EEEENS0_17mutable_buffers_1EN + 0xe0 (0xb3ed1465) [/opt/usr/apps/org.tizen.other/bin/other] + 0x54465
47: _ZN5boost4asio6detail7binder2INS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_serviceIS6_EEEENS0_17mutabl + 0x22 (0xb3ee6d03) [/opt/usr/apps/org.tizen.other/bin/other] + 0x69d03
48: _ZN5boost4asio19asio_handler_invokeINS0_6detail7binder2INS2_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_s + 0xc (0xb3ee5af9) [/opt/usr/apps/org.tizen.other/bin/other] + 0x68af9
49: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2INS3_7read_opINS2_19basic_stream_socketINS2_2ip3tcpENS2 + 0x1a (0xb3ee482f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6782f
50: _ZN5boost4asio6detail19asio_handler_invokeINS1_7binder2INS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21stream_socket_s + 0x16 (0xb3ee33db) [/opt/usr/apps/org.tizen.other/bin/other] + 0x663db
51: _ZN33boost_asio_handler_invoke_helpers6invokeIN5boost4asio6detail7binder2INS3_7read_opINS2_19basic_stream_socketINS2_2ip3tcpENS2 + 0x1a (0xb3ee237b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x6537b
52: _ZN5boost4asio6detail23reactive_socket_recv_opINS0_17mutable_buffers_1ENS1_7read_opINS0_19basic_stream_socketINS0_2ip3tcpENS0_21 + 0x7a (0xb3ee0e43) [/opt/usr/apps/org.tizen.other/bin/other] + 0x63e43
53: boost::asio::detail::task_io_service_operation::complete(boost::asio::detail::task_io_service&, boost::system::error_code const&, unsigned int) + 0x1c (0xb3e95bf1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x18bf1
54: _ZN5boost4asio6detail13epoll_reactor16descriptor_state11do_completeEPNS1_15task_io_serviceEPNS1_25task_io_service_operationERKNS + 0x3a (0xb3e9732b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1a32b
55: boost::asio::detail::task_io_service_operation::complete(boost::asio::detail::task_io_service&, boost::system::error_code const&, unsigned int) + 0x1c (0xb3e95bf1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x18bf1
56: _ZN5boost4asio6detail15task_io_service10do_run_oneERNS1_11scoped_lockINS1_11posix_mutexEEERNS1_27task_io_service_thread_infoERKN + 0x116 (0xb3e97ba3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1aba3
57: boost::asio::detail::task_io_service::run(boost::system::error_code&) + 0xca (0xb3e97773) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1a773
58: boost::asio::io_service::run() + 0x22 (0xb3e97e1b) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1ae1b
59: websocketpp::transport::asio::endpoint<websocketpp::config::debug_asio::transport_config>::run() + 0x12 (0xb3ea3613) [/opt/usr/apps/org.tizen.other/bin/other] + 0x26613
60: sio::client::impl::run_loop() + 0x12 (0xb3e8b043) [/opt/usr/apps/org.tizen.other/bin/other] + 0xe043
61: _ZNKSt7_Mem_fnIMN3sio6client4implEFvvEEclIJEvEEvPS2_DpOT_ + 0x3e (0xb3ef20bf) [/opt/usr/apps/org.tizen.other/bin/other] + 0x750bf
62: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS3_EE6__callIvJEJLj0EEEET_OSt5tupleIJDpT0_EESt12_Index_tupleIJXspT1_EEE + 0x34 (0xb3ef1e11) [/opt/usr/apps/org.tizen.other/bin/other] + 0x74e11
63: _ZNSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS3_EEclIJEvEET0_DpOT_ + 0x1a (0xb3ef1b2f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x74b2f
64: _ZNSt12_Bind_simpleIFSt5_BindIFSt7_Mem_fnIMN3sio6client4implEFvvEEPS4_EEvEE9_M_invokeIJEEEvSt12_Index_tupleIJXspT_EEE + 0x22 (0xb3ef1227) [/opt/usr/apps/org.tizen.other/bin/other] + 0x74227
65: std::_Bind_simple<std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)> ()>::operator()() + 0x10 (0xb3ef06c1) [/opt/usr/apps/org.tizen.other/bin/other] + 0x736c1
66: std::thread::_Impl<std::_Bind_simple<std::_Bind<std::_Mem_fn<void (sio::client::impl::*)()> (sio::client::impl*)> ()> >::_M_run() + 0x12 (0xb3eeeda3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x71da3
67: (0xb5c127dd) [/usr/lib/libstdc++.so.6] + 0x517dd
68: (0xb6f6dd30) [/lib/libpthread.so.0] + 0x6d30
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
t_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.185+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.195+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.195+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.205+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.210+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.210+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.220+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.225+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.225+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.235+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.245+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.245+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.255+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.260+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.260+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.270+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.280+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.280+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.290+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.295+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.295+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.305+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.310+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.310+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.320+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.330+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.330+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.340+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.345+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.345+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.355+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.365+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.365+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.370+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.380+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.380+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.390+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.395+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.395+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.415+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.415+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.415+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.425+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.430+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.430+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.440+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.445+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.445+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.455+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.465+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.465+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.475+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.480+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.480+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.490+0900 D/APP_CORE( 2251): appcore-efl.c: __do_app(470) > [APP 2251] Event: MEM_FLUSH State: PAUSED
04-13 20:27:23.490+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.500+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.500+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.505+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.515+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.515+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.525+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.530+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.530+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.540+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.550+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.550+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.560+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.565+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.565+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.585+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.590+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.590+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.600+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.600+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.600+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.610+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.615+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.615+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.625+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.635+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.635+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.640+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.650+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.650+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.660+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.665+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.665+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.675+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.685+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.685+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.690+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.700+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.700+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.710+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.715+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.715+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.725+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.735+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.735+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.745+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.750+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.750+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.760+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.770+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.770+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.780+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.785+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.785+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.795+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.800+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.800+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.810+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.820+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.820+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.830+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.835+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.835+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.845+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.855+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.855+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.860+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.870+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.870+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.880+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.885+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.885+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.895+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.905+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.905+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.910+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.920+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.920+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.930+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.935+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.935+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.945+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.955+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.955+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.965+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.970+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.970+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.980+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:23.990+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:23.990+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:23.995+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.005+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.005+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.015+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.020+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.020+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.030+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.040+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.040+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.050+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.055+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.055+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.065+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.070+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.070+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.080+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.090+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.090+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.100+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.105+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.105+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.115+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.125+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.125+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.130+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.140+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.140+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.150+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.155+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.155+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.165+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.175+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.175+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.185+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.190+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.190+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.200+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.205+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.205+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.215+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.225+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.225+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.235+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.240+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.240+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.250+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.260+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.260+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.265+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.275+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.275+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.285+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.290+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.290+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.300+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.310+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.310+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.320+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.325+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.325+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.335+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.345+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.345+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.350+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.360+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.360+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.370+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.375+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.375+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.385+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.395+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.395+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.400+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.410+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.410+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.420+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.425+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.425+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.435+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.445+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.445+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.455+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.460+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.460+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.470+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.480+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.480+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.485+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.505+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.505+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.515+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.520+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.520+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.530+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.530+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.530+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.540+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.545+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.545+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.555+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.560+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.565+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.570+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.580+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.580+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.590+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.595+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.595+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.605+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.615+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.615+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.620+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.630+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.630+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.640+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.645+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.645+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.655+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.665+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.665+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.675+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.680+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.680+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.690+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.695+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.695+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.705+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.715+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.715+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.725+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.730+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.730+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.740+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.750+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.750+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.755+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.765+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.765+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.775+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.780+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.780+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.790+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.800+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] NOT_SUPPORTED_FORMAT(0xfe6e0001)
04-13 20:27:24.800+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_calculate_buffer_size] INVALID_PARAMETER(0xffffffea)
04-13 20:27:24.810+0900 D/CAPI_MEDIA_IMAGE_UTIL( 3913): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-13 20:27:24.860+0900 W/CRASH_MANAGER( 3924): worker.c: worker_job(1189) > 11039136f7468142892443
