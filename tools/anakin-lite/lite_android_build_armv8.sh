#!/bin/bash
# This script shows how one can build a anakin for the Android platform using android-tool-chain. 
export ANDROID_NDK=/Users/chenjiao04/Documents/android-ndk-r14b
ANAKIN_LITE_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
echo "-- Anakin lite root dir is: $ANAKIN_LITE_ROOT"

if [ -z "$ANDROID_NDK" ]; then
    echo "-- Did you set ANDROID_NDK variable?"
    exit 1
fi

if [ -d "$ANDROID_NDK" ]; then
    echo "-- Using Android ndk at $ANDROID_NDK"
else
    echo "-- Cannot find ndk: did you install it under $ANDROID_NDK ?"
    exit 1
fi

# build the target into build_android.
BUILD_ROOT=$ANAKIN_LITE_ROOT/build-android-v8

#if [ -d $BUILD_ROOT ];then
#	rm -rf $BUILD_ROOT
#fi

mkdir -p $BUILD_ROOT
echo "-- Build anakin lite Android into: $BUILD_ROOT"

# Now, actually build the android target.
echo "-- Building anakin lite ..."
cd $BUILD_ROOT

#rm -rf *
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../../cmake/android/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_ABI="arm64-v8a" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ARMV8=YES \
	-DANDROID_NATIVE_API_LEVEL=21 \
	-DUSE_ANDROID=YES \
	-DTARGET_IOS=NO \
    -DUSE_OPENMP=YES \
    -DBUILD_LITE_UNIT_TEST=YES \
    -DUSE_OPENCV=NO \
    -DENABLE_OP_TIMER=YES

# build target lib or unit test.
if [ "$(uname)" = 'Darwin' ]; then
    make "-j$(sysctl -n hw.ncpu)" && make install
else
    make "-j$(nproc)" && make install
fi

