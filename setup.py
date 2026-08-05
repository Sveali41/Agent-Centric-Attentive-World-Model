from setuptools import find_namespace_packages, setup


setup(
    name="agent-centric-attentive-world-model",
    version="0.1.0",
    description="Agent-Centric Attentive World Model and reusable domain adapters",
    packages=find_namespace_packages(
        include=["modelBased*", "domain*", "generator*"]
    ),
    include_package_data=True,
)
