cmake_minimum_required(VERSION 2.6)
project(PotholeDetection)
set(PROJECT_VERSION 0.0.2a)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# Where to find CMake modules and OpenCV
if ("$ENV{OPENCV}" STREQUAL "")
    message("ERROR: The OPENCV environmental/global variable is not set or is empty")
    return(0)
elseif("$ENV{RAPIDJSON}" STREQUAL "")
    message("ERROR: The RAPIDJSON environmental/global variable is not set or is empty")
    return(0)
else ()

    message("$ENV{OPENCV}")
    message("$ENV{RAPIDJSON}")

    set(OPENCV_DIR $ENV{OPENCV}/install)
    set(RAPIDJSON_INCLUDE $ENV{RAPIDJSON}/include)
    set(PHD_INCLUDE ${PROJECT_SOURCE_DIR}/include)

    if(CMAKE_SYSTEM_NAME STREQUAL "windows")
        set(OPENCV_LIBS "${OPENCV_DIR}/x64/mingw/lib/libopencv_world340.dll.a")
    else ()
        find_package(OpenCv REQUIRED)
    endif(CMAKE_SYSTEM_NAME STREQUAL "windows")

#    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-modules")

    find_package(OPENCV REQUIRED)

    include_directories(
            ${OpenCV_INCLUDE_DIRS}
            ${RAPIDJSON_INCLUDE}
            ${PHD_INCLUDE}
    )

    # Create Library
    add_library(PotholeDetectionLib SHARED
            src/core.cpp #include/phdetection/core.hpp
            src/histogram.cpp #include/phdetection/histogram.hpp
            src/segmentation.cpp #include/phdetection/segmentation.hpp
            src/math.cpp #include/phdetection/math.hpp
            src/io.cpp #include/phdetection/io.hpp
            src/features_extraction.cpp #include/phdetection/features_extraction.hpp
            src/svm.cpp #include/phdetection/svm.hpp
            src/ml_utils.cpp #include/phdetection/ml_utils.hpp
            src/superpixeling.cpp #include/phdetection/superpixeling.hpp
            src/ontologies.cpp #include/phdetection/ontologies.hpp
            src/hog.cpp #include/phdetection/hog.hpp
            src/bayes.cpp #include/phdetection/bayes.hpp
            src/fe_utils.cpp #include/phdetection/fe_utils.hpp
            #include/phdetection/optional.hpp
    )

    set(PUBLIC_HEADERS
            include/phdetection/core.hpp;
            include/phdetection/histogram.hpp;
            include/phdetection/segmentation.hpp;
            include/phdetection/math.hpp;
            include/phdetection/io.hpp;
            include/phdetection/features_extraction.hpp;
            include/phdetection/svm.hpp;
            include/phdetection/ml_utils.hpp;
            include/phdetection/superpixeling.hpp;
            include/phdetection/ontologies.hpp;
            include/phdetection/hog.hpp;
            include/phdetection/bayes.hpp;
            include/phdetection/fe_utils.hpp;
            include/phdetection/optional.hpp;
    )

    set_target_properties(PotholeDetectionLib PROPERTIES
            VERSION ${PROJECT_VERSION}
            PREFIX "lib"
            OUTPUT_NAME "phd_world"
            PUBLIC_HEADER "${PUBLIC_HEADERS}"
    )

    # linking library with opencv
    target_link_libraries(PotholeDetectionLib PUBLIC ${OpenCV_LIBS})

    # create executable binary
    add_executable(PotholeDetection main.cpp)

    target_link_libraries(PotholeDetection PUBLIC PotholeDetectionLib)

    install(TARGETS         PotholeDetectionLib PotholeDetection
            ARCHIVE         DESTINATION         "lib/static"
            LIBRARY         DESTINATION         "lib/shared"
            RUNTIME         DESTINATION         "bin"
            PUBLIC_HEADER   DESTINATION         "include/phdetection"
    )

    return(1)
endif ()