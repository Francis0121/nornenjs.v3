
var os = require('os');
var interfaces = os.networkInterfaces();

function getIpAddress() {

    var ipAddress = null;

    for (var k in interfaces) {
        for (var k2 in interfaces[k]) {
            var address = interfaces[k][k2];
            if (address.family === 'IPv4' && !address.internal && ipAddress == null) {
                ipAddress = address.address;
            }
        }
    }

    return ipAddress;
}

module.exports.getIpAddress = getIpAddress;
