# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(FHELRModelCreditCardFraud VERSION 1.0.0 LANGUAGES CXX)

# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FHELRModelCreditCardFraud_SOURCE_DIR}/../bin)

add_executable(fhelrmodel examples.cpp)
target_sources(fhelrmodel
    PRIVATE
        1_bfv_basics.cpp
        2_encoders.cpp
        3_levels.cpp
        4_ckks_basics.cpp
        5_rotation.cpp
        6_performance.cpp
		model.cpp
)

# Import Microsoft SEAL
find_package(SEAL 3.4.5 EXACT REQUIRED)

# Link Microsoft SEAL
target_link_libraries(fhelrmodel SEAL::seal)
