check_include_file_cxx(stdint.h HAVE_STDINT_H)
if(HAVE_STDINT_H)
    add_definitions(-DHAVE_STDINT_H)
endif()

build_lib(
    LIBNAME oracle-wifi-manager
    SOURCE_FILES model/oracle-wifi-manager.cc
    HEADER_FILES model/oracle-wifi-manager.h
    LIBRARIES_TO_LINK
        ${libcore}
        ${libwifi}
)
