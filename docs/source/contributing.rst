Contributing to Surfacia
========================

We welcome contributions to Surfacia! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the Repository**
   
   Fork the Surfacia repository on GitHub and clone your fork:
   
   .. code-block:: bash
   
      git clone https://github.com/yourusername/surfacia.git
      cd surfacia

2. **Set Up Development Environment**
   
   Create a virtual environment and install development dependencies:
   
   .. code-block:: bash
   
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e .[dev]

3. **Install Pre-commit Hooks**
   
   .. code-block:: bash
   
      pre-commit install

Types of Contributions
----------------------

Code Contributions
~~~~~~~~~~~~~~~~~~

* **Bug Fixes**: Fix existing issues or bugs
* **New Features**: Implement new functionality
* **Performance Improvements**: Optimize existing code
* **Code Refactoring**: Improve code structure and maintainability

Documentation Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **API Documentation**: Improve docstrings and API references
* **User Guides**: Create or improve user-facing documentation
* **Tutorials**: Develop step-by-step tutorials
* **Examples**: Add practical usage examples

Testing Contributions
~~~~~~~~~~~~~~~~~~~~~

* **Unit Tests**: Add tests for individual functions and classes
* **Integration Tests**: Test complete workflows and interactions
* **Performance Tests**: Benchmark and performance validation
* **Test Data**: Contribute test datasets and reference results

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

* Follow PEP 8 style guidelines
* Use type hints for function parameters and return values
* Write clear, descriptive variable and function names
* Keep functions focused and modular

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

* Write comprehensive docstrings for all public functions and classes
* Use Google-style docstrings
* Include examples in docstrings where appropriate
* Update documentation when adding new features

Testing Requirements
~~~~~~~~~~~~~~~~~~~~

* Write tests for all new functionality
* Ensure existing tests pass
* Aim for high test coverage
* Include both positive and negative test cases

Commit Guidelines
~~~~~~~~~~~~~~~~~

* Write clear, descriptive commit messages
* Use conventional commit format when possible
* Keep commits focused on a single change
* Reference relevant issues in commit messages

Pull Request Process
--------------------

1. **Create a Branch**
   
   Create a feature branch for your changes:
   
   .. code-block:: bash
   
      git checkout -b feature/your-feature-name

2. **Make Changes**
   
   Implement your changes following the development guidelines above.

3. **Test Your Changes**
   
   Run the test suite to ensure your changes don't break existing functionality:
   
   .. code-block:: bash
   
      pytest tests/
      python -m surfacia --help  # Test CLI functionality

4. **Update Documentation**
   
   Update relevant documentation and ensure it builds correctly:
   
   .. code-block:: bash
   
      cd docs/
      make html

5. **Submit Pull Request**
   
   Push your branch and create a pull request:
   
   .. code-block:: bash
   
      git push origin feature/your-feature-name

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

Before submitting your pull request, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Pull request description explains the changes

Code Review Process
-------------------

All contributions go through a code review process:

1. **Automated Checks**: CI/CD pipeline runs automated tests and checks
2. **Peer Review**: Project maintainers review the code for quality and correctness
3. **Feedback**: Reviewers may request changes or improvements
4. **Approval**: Once approved, the pull request will be merged

Review Criteria
~~~~~~~~~~~~~~~

Reviewers will evaluate:

* **Correctness**: Does the code work as intended?
* **Quality**: Is the code well-written and maintainable?
* **Testing**: Are there adequate tests for the changes?
* **Documentation**: Is the documentation complete and accurate?
* **Compatibility**: Does it maintain backward compatibility?

Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* Clear description of the issue
* Steps to reproduce the problem
* Expected vs. actual behavior
* System information (OS, Python version, etc.)
* Relevant error messages or logs

Feature Requests
~~~~~~~~~~~~~~~~

For feature requests, please provide:

* Clear description of the desired functionality
* Use cases and motivation
* Potential implementation approach
* Examples of similar features in other tools

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

* Be respectful and considerate in all interactions
* Welcome newcomers and help them get started
* Focus on constructive feedback and collaboration
* Respect different viewpoints and experiences

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For general questions and community discussion
* **Pull Requests**: For code contributions and reviews
* **Documentation**: For documentation improvements and clarifications

Recognition
-----------

Contributors are recognized in several ways:

* **Contributors List**: All contributors are listed in the project README
* **Release Notes**: Significant contributions are highlighted in release notes
* **Documentation**: Contributors are credited in relevant documentation sections
* **Community**: Active contributors may be invited to join the core team

Getting Help
------------

If you need help with contributing:

* Check existing issues and documentation
* Ask questions in GitHub Discussions
* Reach out to maintainers for guidance
* Join community meetings and discussions

Thank you for contributing to Surfacia! Your contributions help make molecular surface analysis more accessible and powerful for the scientific community.