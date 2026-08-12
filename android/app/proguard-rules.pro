# Keep MediaPipe solution core classes
-keep class com.google.mediapipe.** { *; }

# Keep Protobuf classes used by MediaPipe
-keep class com.google.protobuf.** { *; }
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite {
    <fields>;
}

# Keep common Flogger rules needed with MediaPipe
-keep class com.google.common.flogger.** { *; }

# Keep TensorFlow Lite and LiteRT classes
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# Avoid warnings for missing optional dependencies in MediaPipe / Protobuf
-dontwarn com.google.protobuf.**
-dontwarn javax.annotation.**
-dontwarn com.google.auto.value.extension.memoized.**
-dontwarn com.google.mediapipe.proto.**
-dontwarn org.checkerframework.**
