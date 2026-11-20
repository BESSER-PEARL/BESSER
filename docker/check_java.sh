#!/bin/sh
# Fails the image build when the installed Java is affected by CVE-2026-114938
# ("Java4Shell", the AVX-512 Base64 intrinsic) or when the JVM does not apply
# the mitigation flags in JAVA_TOOL_OPTIONS (see the Dockerfile).
# Fixed releases: OpenJDK 17.0.21.1, 21.0.13.1, 25.0.5.1, 27.0.1.1.
set -eu

fail() { echo "check_java: $*" >&2; exit 1; }

version="$(java -XshowSettings:properties -version 2>&1 | sed -n 's/^ *java\.version = //p')"
case "$version" in
    17.*) fixed=17.0.21.1 ;;
    21.*) fixed=21.0.13.1 ;;
    25.*) fixed=25.0.5.1 ;;
    27.*) fixed=27.0.1.1 ;;
    *) fail "no known fixed release for Java '$version'" ;;
esac
dpkg --compare-versions "$version" ge "$fixed" \
    || fail "Java $version is affected by CVE-2026-114938 (fixed in $fixed)"

flags="$(java -XX:+PrintFlagsFinal -version 2>/dev/null)"
flag() { printf '%s\n' "$flags" | awk -v name="$1" '$2 == name { print $4 }'; }
[ "$(flag UseBASE64Intrinsics)" = false ] || fail "UseBASE64Intrinsics is not off"
# UseAVX exists only on x86-64; 2 keeps the JIT off AVX-512.
if [ "$(uname -m)" = x86_64 ]; then
    avx="$(flag UseAVX)"
    [ -n "$avx" ] && [ "$avx" -le 2 ] || fail "UseAVX is '$avx', not at most 2"
fi

echo "check_java: Java $version (fixed in $fixed), JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:-}"
