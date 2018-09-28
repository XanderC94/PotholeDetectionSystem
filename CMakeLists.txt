cmake_minimum_required(VERSION 3.1)
project(PotholeDetection  VERSION 0.0.2)

EXPORT(PACKAGE ${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# Where to find CMake modules and OpenCV
if ("$ENV{OpenCV_DIR}" STREQUAL "")
    message("ERROR: The Open Computer Vision environmental/global variable is not set or is empty")
    return(0)
else ()

    message("Open Computer Vision: $ENV{OpenCV_DIR}")

    set(RapidJSON_DIR "lib/RapidJSON")

#    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-modules")

    find_package(OpenCV REQUIRED)
#    find_package(RapidJSON REQUIRED)

    include_directories(
            ${OpenCV_INCLUDE_DIRS}
            ${RapidJSON_DIR} #/include
            ${PROJECT_SOURCE_DIR}/include
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

    CONFIGURE_FILE(
            ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY
    )

    CONFIGURE_FILE(
            ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
            ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}Config.cmake @ONLY
    )

    return(1)
endif ()