from os import environ as ENV_VARS
import platform

OK = 0
ERR = 1

# system
PLATFORM = platform.system()
WIN = PLATFORM == "Windows"
OSX = PLATFORM == "Darwin"
CI = ENV_VARS.get("JENKINS_HOME")

# installer versions
SBT_VERSION = ENV_VARS.get("SBT_VERSION", "1.2.8")
SYSML2_API_RELEASE = ENV_VARS.get("SYSML2_API_RELEASE", "2021-06")
SYSML2_RELEASE = ENV_VARS.get("SYSML2_RELEASE", "2021-06.1")

# API Configuration
API_PORT = ENV_VARS.get("API_PORT", 9000)
LOCAL_API_SERVER_URL = ENV_VARS.get("LOCAL_API_SERVER_URL", "http://localhost")
REMOTE_API_SERVER_URL = ENV_VARS.get("REMOTE_API_SERVER_URL", "http://sysml2.intercax.com")
SBT_GITHUB = "https://github.com/sbt/sbt"
SYSML_RELEASE_GITHUB = "https://github.com/Systems-Modeling/SysML-v2-Release"
SYSML_API_GITHUB = "https://github.com/Systems-Modeling/SysML-v2-API-Services"
