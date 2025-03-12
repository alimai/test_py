################################################################################
# Define the target name
################################################################################
set(TargetName ODSChecker)

################################################################################
# Source groups
################################################################################
AddSourceGroups()
   
################################################################################
# Target
################################################################################
add_executable(${TargetName} ${ALL_TARGET_FILES})

################################################################################
# Set Target output name
################################################################################
PopSetTargetOutputName()

################################################################################
# Output directory
################################################################################
PopSetTargetOutputDirectoryToDeploy()
    
################################################################################
# Compile Options
################################################################################
include(${TEMP_CAMKE_FILES_DIR}/Configurations/CompileOptions.cmake)
    
################################################################################
# Compile Definitions
################################################################################
include(${TEMP_CAMKE_FILES_DIR}/Configurations/CompileDefinitions.cmake)

################################################################################
# Add Extra compile definitions
################################################################################
 
################################################################################
# Link Options
################################################################################
include(${TEMP_CAMKE_FILES_DIR}/Configurations/LinkOptions.cmake)
    
########################################
#Define Target Extra Includes
########################################
if (LITESHAREDUTILS_INCLUDE_DIR) #*******		
	target_include_directories(${TargetName} PUBLIC ${LITESHAREDUTILS_INCLUDE_DIR}) 
endif()	

################################################################################
#Define Dependency Includes and Links
################################################################################
AddTargetDependency(PopBaseCore "BaseCore;BasePublic") #*******
AddTargetDependency(PopModel "Model") #*******
AddTargetDependency(PopAttachmentLibrary "AttachmentLibrary") #*******
AddTargetDependency(PopPlanScheme "AlignScheme;IO_Base;IO_DDM;IO_ODS") #*******
AddTargetDependency(PopNetWork "NetworkBase;OnlineActivation") #*******
if("${PlatformTarget}" STREQUAL "x86")
	AddTargetThirdPartyDependencyWithConfig(boost Debug "libboost_program_options-vc140-mt-gd-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost NOptRelease_Debug "libboost_program_options-vc140-mt-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release_Debug "libboost_program_options-vc140-mt-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release "libboost_program_options-vc140-mt-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Debug "libboost_filesystem-vc140-mt-gd-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost NOptRelease_Debug "libboost_filesystem-vc140-mt-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release_Debug "libboost_filesystem-vc140-mt-x32-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release "libboost_filesystem-vc140-mt-x32-1_80")
elseif("${PlatformTarget}" STREQUAL "x64")
	AddTargetThirdPartyDependencyWithConfig(boost Debug "libboost_program_options-vc140-mt-gd-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost NOptRelease_Debug "libboost_program_options-vc140-mt-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release_Debug "libboost_program_options-vc140-mt-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release "libboost_program_options-vc140-mt-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Debug "libboost_program_options-vc140-mt-gd-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost NOptRelease_Debug "libboost_program_options-vc140-mt-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release_Debug "libboost_program_options-vc140-mt-x64-1_80")
	AddTargetThirdPartyDependencyWithConfig(boost Release "libboost_program_options-vc140-mt-x64-1_80")
endif()

################################################################################
#Define Dependency in the same project
################################################################################
AddIntrinsicTargetDependencies("CommonUtil;AttachmentLibraryAgent") #*******

################################################################################
#Define the thrid party dependencies
################################################################################

################################################################################
# Add the prepare install
################################################################################
PopAddExecutableTargetInstallPrepare()
PopAddExecutableTargetPackage()