"""
Production Platform Compatibility Manager

A comprehensive cross-platform compatibility layer that ensures reliable operation
across Docker, Kubernetes, bare metal, Windows, Linux, and macOS environments.

Features:
- Automatic platform detection
- Intelligent directory management
- Database connection handling
- Environment-specific configuration
- Comprehensive logging setup
- Fallback strategies for all resources

Author: Production Team
Version: 2.0.0
"""

import os
import sys
import platform
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import socket

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Supported platform types."""
    LINUX = "Linux"
    WINDOWS = "Windows"
    DARWIN = "Darwin"  # macOS
    UNKNOWN = "Unknown"


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    BARE_METAL = "bare_metal"
    DEVELOPMENT = "development"


@dataclass
class DirectoryConfig:
    """Configuration for directory paths."""
    # Primary directories
    data_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    
    # Application-specific
    uploads_dir: Optional[Path] = None
    outputs_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    
    # Flags
    use_temp_fallback: bool = True
    create_subdirs: bool = True


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "app_db"
    user: str = "app_user"
    password: str = "app_password"
    
    # Connection options
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # SSL options
    ssl_mode: str = "prefer"
    ssl_cert: Optional[Path] = None
    
    def get_connection_string(self, driver: str = "postgresql") -> str:
        """
        Build database connection string.
        
        Args:
            driver: Database driver (postgresql, mysql, etc.)
        
        Returns:
            Connection string
        """
        if driver == "postgresql":
            return (
                f"postgresql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
            )
        elif driver == "mysql":
            return (
                f"mysql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
            )
        else:
            raise ValueError(f"Unsupported driver: {driver}")
    
    def get_async_connection_string(self, driver: str = "postgresql") -> str:
        """Build async database connection string."""
        if driver == "postgresql":
            return (
                f"postgresql+asyncpg://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
            )
        else:
            raise ValueError(f"Unsupported async driver: {driver}")


class PlatformManager:
    """
    Production-grade platform compatibility manager.
    
    Automatically detects the deployment environment and provides
    platform-specific configurations for directories, databases,
    logging, and other resources.
    
    Features:
    - Automatic platform and environment detection
    - Intelligent directory creation with fallbacks
    - Database connection string generation
    - Environment-specific logging configuration
    - Resource availability checking
    - Comprehensive error handling
    
    Usage:
        manager = PlatformManager()
        data_dir = manager.get_data_directory()
        db_config = manager.get_database_config()
        manager.configure_logging()
    """
    
    def __init__(self, app_name: str = "app"):
        """
        Initialize the platform manager.
        
        Args:
            app_name: Application name for directory paths
        """
        self.app_name = app_name
        self.platform = self._detect_platform()
        self.environment = self._detect_environment()
        
        # Cache
        self._directory_cache: Dict[str, Path] = {}
        self._capabilities: Dict[str, bool] = {}
        
        logger.info(f"🖥️  Platform: {self.platform.value}")
        logger.info(f"🌍 Environment: {self.environment.value}")
        logger.info(f"📦 App Name: {self.app_name}")
    
    # Platform Detection
    
    def _detect_platform(self) -> PlatformType:
        """
        Detect the operating system platform.
        
        Returns:
            PlatformType enum value
        """
        system = platform.system()
        
        try:
            return PlatformType(system)
        except ValueError:
            logger.warning(f"Unknown platform: {system}")
            return PlatformType.UNKNOWN
    
    def _detect_environment(self) -> DeploymentEnvironment:
        """
        Detect the deployment environment.
        
        Returns:
            DeploymentEnvironment enum value
        """
        # Check Kubernetes first (more specific)
        if self._is_kubernetes():
            return DeploymentEnvironment.KUBERNETES
        
        # Check Docker
        if self._is_docker():
            return DeploymentEnvironment.DOCKER
        
        # Check if development (common indicators)
        if self._is_development():
            return DeploymentEnvironment.DEVELOPMENT
        
        # Default to bare metal
        return DeploymentEnvironment.BARE_METAL
    
    def _is_docker(self) -> bool:
        """
        Detect if running in Docker container.
        
        Returns:
            True if running in Docker
        """
        # Method 1: Check for .dockerenv file
        if Path('/.dockerenv').exists():
            return True
        
        # Method 2: Check cgroup
        try:
            with open('/proc/1/cgroup', 'r') as f:
                content = f.read()
                if 'docker' in content or 'containerd' in content:
                    return True
        except (FileNotFoundError, PermissionError):
            pass
        
        # Method 3: Check for Docker environment variables
        docker_vars = ['DOCKER_CONTAINER', 'DOCKER_HOST']
        if any(var in os.environ for var in docker_vars):
            return True
        
        return False
    
    def _is_kubernetes(self) -> bool:
        """
        Detect if running in Kubernetes.
        
        Returns:
            True if running in Kubernetes
        """
        # Method 1: Kubernetes service environment variables
        k8s_vars = [
            'KUBERNETES_SERVICE_HOST',
            'KUBERNETES_SERVICE_PORT',
            'KUBERNETES_PORT'
        ]
        if any(var in os.environ for var in k8s_vars):
            return True
        
        # Method 2: Kubernetes secrets mount
        k8s_paths = [
            Path('/var/run/secrets/kubernetes.io'),
            Path('/var/run/secrets/kubernetes.io/serviceaccount')
        ]
        if any(path.exists() for path in k8s_paths):
            return True
        
        return False
    
    def _is_development(self) -> bool:
        """
        Detect if running in development environment.
        
        Returns:
            True if development environment
        """
        # Check common development indicators
        dev_indicators = [
            os.environ.get('ENVIRONMENT') == 'development',
            os.environ.get('ENV') == 'dev',
            os.environ.get('DEBUG') == 'true',
            os.environ.get('FLASK_ENV') == 'development',
            os.environ.get('NODE_ENV') == 'development',
        ]
        
        return any(dev_indicators)
    
    # Directory Management
    
    def get_data_directory(self, create: bool = True) -> Path:
        """
        Get platform-appropriate data directory with intelligent fallbacks.
        
        Priority order:
        1. Explicit DATA_DIR environment variable
        2. Docker volume mount (/app/data)
        3. Kubernetes PersistentVolume (/data)
        4. User home directory (~/.{app_name}/data)
        5. System temp directory (last resort)
        
        Args:
            create: Whether to create directory if it doesn't exist
        
        Returns:
            Path to data directory
        """
        cache_key = "data_dir"
        if cache_key in self._directory_cache:
            return self._directory_cache[cache_key]
        
        candidates = self._get_data_directory_candidates()
        
        for candidate in candidates:
            if self._ensure_directory_writable(candidate, create=create):
                logger.info(f"✅ Data directory: {candidate}")
                self._directory_cache[cache_key] = candidate
                return candidate
        
        # Should never reach here due to temp fallback
        raise RuntimeError("Could not establish writable data directory")
    
    def _get_data_directory_candidates(self) -> List[Path]:
        """Get ordered list of data directory candidates."""
        candidates = []
        
        # Priority 1: Explicit configuration
        if 'DATA_DIR' in os.environ:
            candidates.append(Path(os.environ['DATA_DIR']))
        
        # Priority 2: Docker volume
        if self.environment == DeploymentEnvironment.DOCKER:
            candidates.extend([
                Path('/app/data'),
                Path('/data'),
            ])
        
        # Priority 3: Kubernetes PersistentVolume
        if self.environment == DeploymentEnvironment.KUBERNETES:
            candidates.extend([
                Path('/data'),
                Path(f'/var/lib/{self.app_name}'),
            ])
        
        # Priority 4: User home directory
        try:
            home = Path.home()
            candidates.extend([
                home / f'.{self.app_name}' / 'data',
                home / self.app_name / 'data',
            ])
        except Exception:
            pass
        
        # Priority 5: System locations (Unix-like)
        if self.platform != PlatformType.WINDOWS:
            candidates.extend([
                Path(f'/var/lib/{self.app_name}'),
                Path(f'/opt/{self.app_name}/data'),
            ])
        
        # Priority 6: Windows AppData
        if self.platform == PlatformType.WINDOWS:
            appdata = os.environ.get('APPDATA')
            if appdata:
                candidates.append(Path(appdata) / self.app_name / 'data')
        
        # Priority 7: System temp (last resort)
        candidates.append(
            Path(tempfile.gettempdir()) / self.app_name / 'data'
        )
        
        return candidates
    
    def get_temp_directory(self, create: bool = True) -> Path:
        """
        Get platform-appropriate temp directory.
        
        Args:
            create: Whether to create directory if it doesn't exist
        
        Returns:
            Path to temp directory
        """
        cache_key = "temp_dir"
        if cache_key in self._directory_cache:
            return self._directory_cache[cache_key]
        
        # Use system temp directory
        temp_base = Path(tempfile.gettempdir())
        temp_dir = temp_base / self.app_name / 'temp'
        
        if self._ensure_directory_writable(temp_dir, create=create):
            self._directory_cache[cache_key] = temp_dir
            return temp_dir
        
        # Fallback to system temp root
        return temp_base
    
    def get_logs_directory(self, create: bool = True) -> Path:
        """
        Get platform-appropriate logs directory.
        
        Args:
            create: Whether to create directory if it doesn't exist
        
        Returns:
            Path to logs directory
        """
        cache_key = "logs_dir"
        if cache_key in self._directory_cache:
            return self._directory_cache[cache_key]
        
        candidates = []
        
        # Priority 1: Explicit configuration
        if 'LOG_DIR' in os.environ:
            candidates.append(Path(os.environ['LOG_DIR']))
        
        # Priority 2: Standard log locations
        if self.environment in [DeploymentEnvironment.DOCKER, DeploymentEnvironment.KUBERNETES]:
            # Containers typically log to stdout, but we can still write files
            candidates.extend([
                Path(f'/var/log/{self.app_name}'),
                Path('/var/log'),
            ])
        
        # Priority 3: Application data directory
        try:
            data_dir = self.get_data_directory(create=False)
            candidates.append(data_dir / 'logs')
        except Exception:
            pass
        
        # Priority 4: Temp directory
        candidates.append(self.get_temp_directory() / 'logs')
        
        for candidate in candidates:
            if self._ensure_directory_writable(candidate, create=create):
                self._directory_cache[cache_key] = candidate
                return candidate
        
        # Fallback to temp
        return self.get_temp_directory()
    
    def get_cache_directory(self, create: bool = True) -> Path:
        """
        Get platform-appropriate cache directory.
        
        Args:
            create: Whether to create directory if it doesn't exist
        
        Returns:
            Path to cache directory
        """
        cache_key = "cache_dir"
        if cache_key in self._directory_cache:
            return self._directory_cache[cache_key]
        
        candidates = []
        
        # Priority 1: Explicit configuration
        if 'CACHE_DIR' in os.environ:
            candidates.append(Path(os.environ['CACHE_DIR']))
        
        # Priority 2: Platform-specific cache locations
        if self.platform == PlatformType.LINUX or self.platform == PlatformType.DARWIN:
            try:
                home = Path.home()
                candidates.append(home / '.cache' / self.app_name)
            except Exception:
                pass
        
        if self.platform == PlatformType.WINDOWS:
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                candidates.append(Path(localappdata) / self.app_name / 'cache')
        
        # Priority 3: Data directory
        try:
            data_dir = self.get_data_directory(create=False)
            candidates.append(data_dir / 'cache')
        except Exception:
            pass
        
        # Priority 4: Temp directory
        candidates.append(self.get_temp_directory() / 'cache')
        
        for candidate in candidates:
            if self._ensure_directory_writable(candidate, create=create):
                self._directory_cache[cache_key] = candidate
                return candidate
        
        return self.get_temp_directory()
    
    def get_directory_config(self) -> DirectoryConfig:
        """
        Get comprehensive directory configuration.
        
        Returns:
            DirectoryConfig with all paths
        """
        data_dir = self.get_data_directory()
        
        config = DirectoryConfig(
            data_dir=data_dir,
            temp_dir=self.get_temp_directory(),
            logs_dir=self.get_logs_directory(),
            cache_dir=self.get_cache_directory(),
            uploads_dir=data_dir / 'uploads',
            outputs_dir=data_dir / 'outputs',
            models_dir=data_dir / 'models',
        )
        
        # Create subdirectories if configured
        if config.create_subdirs:
            for subdir in [config.uploads_dir, config.outputs_dir, config.models_dir]:
                if subdir:
                    self._ensure_directory_writable(subdir, create=True)
        
        return config
    
    def _ensure_directory_writable(
        self,
        directory: Path,
        create: bool = True
    ) -> bool:
        """
        Ensure directory exists and is writable.
        
        Args:
            directory: Directory path to check/create
            create: Whether to create if it doesn't exist
        
        Returns:
            True if directory is writable, False otherwise
        """
        try:
            # Create directory if needed
            if create:
                directory.mkdir(parents=True, exist_ok=True)
            elif not directory.exists():
                return False
            
            # Test write permissions
            test_file = directory / f'.write_test_{os.getpid()}'
            test_file.write_text('test')
            test_file.unlink()
            
            # Try to set permissions (may fail in containers)
            try:
                if self.platform != PlatformType.WINDOWS:
                    os.chmod(directory, 0o755)
            except (OSError, PermissionError):
                pass
            
            return True
            
        except (OSError, PermissionError, Exception) as e:
            logger.debug(f"Directory {directory} not writable: {e}")
            return False
    
    # Database Configuration
    
    def get_database_host(self) -> str:
        """
        Get database host with environment-specific discovery.
        
        Returns:
            Database hostname
        """
        # Priority 1: Explicit configuration
        env_vars = ['DATABASE_HOST', 'POSTGRES_HOST', 'DB_HOST']
        for var in env_vars:
            if var in os.environ:
                return os.environ[var]
        
        # Priority 2: Kubernetes service discovery
        if self.environment == DeploymentEnvironment.KUBERNETES:
            # Check for common service names
            k8s_services = [
                'postgres-service',
                'postgresql',
                'postgres',
                'database',
            ]
            
            for service in k8s_services:
                env_var = f"{service.upper().replace('-', '_')}_SERVICE_HOST"
                if env_var in os.environ:
                    return os.environ[env_var]
            
            # Default Kubernetes service name
            return 'postgres-service'
        
        # Priority 3: Docker Compose service discovery
        if self.environment == DeploymentEnvironment.DOCKER:
            # Try common Docker Compose service names
            docker_hosts = ['postgres', 'postgresql', 'database', 'db']
            
            # Check if hostname resolves
            for host in docker_hosts:
                try:
                    socket.gethostbyname(host)
                    return host
                except socket.gaierror:
                    continue
            
            return 'postgres'
        
        # Priority 4: Local development
        return 'localhost'
    
    def get_database_config(
        self,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None
    ) -> DatabaseConfig:
        """
        Get comprehensive database configuration.
        
        Args:
            db_name: Database name override
            db_user: Database user override
        
        Returns:
            DatabaseConfig object
        """
        return DatabaseConfig(
            host=self.get_database_host(),
            port=int(os.getenv('DATABASE_PORT', os.getenv('POSTGRES_PORT', '5432'))),
            database=db_name or os.getenv('DATABASE_NAME', os.getenv('POSTGRES_DB', f'{self.app_name}_db')),
            user=db_user or os.getenv('DATABASE_USER', os.getenv('POSTGRES_USER', f'{self.app_name}_user')),
            password=os.getenv('DATABASE_PASSWORD', os.getenv('POSTGRES_PASSWORD', 'changeme')),
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            ssl_mode=os.getenv('DB_SSL_MODE', 'prefer'),
        )
    
    # Logging Configuration
    
    def configure_logging(
        self,
        level: str = 'INFO',
        log_to_file: bool = True,
        log_format: Optional[str] = None
    ) -> None:
        """
        Configure platform-appropriate logging.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file (in addition to stdout)
            log_format: Custom log format string
        """
        # Get log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Default format
        if log_format is None:
            log_format = (
                '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
            )
        
        # Build handlers
        handlers = []
        
        # Always log to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(stdout_handler)
        
        # File logging for non-container environments
        if log_to_file and self.environment not in [
            DeploymentEnvironment.DOCKER,
            DeploymentEnvironment.KUBERNETES
        ]:
            try:
                log_dir = self.get_logs_directory()
                log_file = log_dir / f'{self.app_name}.log'
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter(log_format))
                handlers.append(file_handler)
                
                logger.info(f"📝 Logging to file: {log_file}")
            except Exception as e:
                logger.warning(f"Could not set up file logging: {e}")
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True  # Override any existing config
        )
        
        logger.info(f"✅ Logging configured: level={level}")
    
    # System Information
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with system details
        """
        info = {
            'platform': {
                'system': self.platform.value,
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            },
            'environment': {
                'type': self.environment.value,
                'is_docker': self.environment == DeploymentEnvironment.DOCKER,
                'is_kubernetes': self.environment == DeploymentEnvironment.KUBERNETES,
                'is_development': self.environment == DeploymentEnvironment.DEVELOPMENT,
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'executable': sys.executable,
            },
            'directories': {
                'data': str(self.get_data_directory()),
                'temp': str(self.get_temp_directory()),
                'logs': str(self.get_logs_directory()),
                'cache': str(self.get_cache_directory()),
            },
            'application': {
                'name': self.app_name,
            }
        }
        
        return info
    
    def check_requirements(self) -> Tuple[bool, List[str]]:
        """
        Check if all system requirements are met.
        
        Returns:
            Tuple of (all_ok, list_of_issues)
        """
        issues = []
        
        # Check directory access
        try:
            self.get_data_directory()
        except Exception as e:
            issues.append(f"Data directory not accessible: {e}")
        
        try:
            self.get_temp_directory()
        except Exception as e:
            issues.append(f"Temp directory not accessible: {e}")
        
        # Check database connectivity (if configured)
        try:
            db_host = self.get_database_host()
            # Try to resolve hostname
            socket.gethostbyname(db_host)
        except socket.gaierror:
            issues.append(f"Cannot resolve database host: {db_host}")
        except Exception as e:
            issues.append(f"Database host check failed: {e}")
        
        # Check Python version
        if sys.version_info < (3, 7):
            issues.append(f"Python 3.7+ required, found {sys.version}")
        
        return (len(issues) == 0, issues)
    
    def print_diagnostics(self) -> None:
        """Print comprehensive diagnostic information."""
        print("\n" + "="*60)
        print("Platform Manager Diagnostics")
        print("="*60)
        
        info = self.get_system_info()
        
        print(f"\n🖥️  Platform:")
        for key, value in info['platform'].items():
            print(f"  {key}: {value}")
        
        print(f"\n🌍 Environment:")
        for key, value in info['environment'].items():
            print(f"  {key}: {value}")
        
        print(f"\n🐍 Python:")
        for key, value in info['python'].items():
            print(f"  {key}: {value}")
        
        print(f"\n📁 Directories:")
        for key, value in info['directories'].items():
            print(f"  {key}: {value}")
        
        print(f"\n🔍 Requirements Check:")
        all_ok, issues = self.check_requirements()
        if all_ok:
            print("  ✅ All requirements met")
        else:
            print("  ❌ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        
        print("\n" + "="*60 + "\n")


# Global singleton instance
_platform_manager: Optional[PlatformManager] = None


def get_platform_manager(app_name: str = "app") -> PlatformManager:
    """
    Get or create global platform manager instance.
    
    Args:
        app_name: Application name
    
    Returns:
        PlatformManager instance
    """
    global _platform_manager
    
    if _platform_manager is None:
        _platform_manager = PlatformManager(app_name)
    
    return _platform_manager


# Convenience functions

def get_data_dir(app_name: str = "app") -> Path:
    """Get data directory."""
    return get_platform_manager(app_name).get_data_directory()


def get_temp_dir(app_name: str = "app") -> Path:
    """Get temp directory."""
    return get_platform_manager(app_name).get_temp_directory()


def get_logs_dir(app_name: str = "app") -> Path:
    """Get logs directory."""
    return get_platform_manager(app_name).get_logs_directory()


def setup_logging(level: str = 'INFO', app_name: str = "app") -> None:
    """Configure logging."""
    get_platform_manager(app_name).configure_logging(level)

platform_manager = PlatformManager()