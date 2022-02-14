## Release notes
---

### 0.9.2 (February 14, 2022)

##### Bug fixes
- Fixed a RNG-related bug that could result in slightly different datasets being generated on Linux and Windows from the same specification.

##### Enhancements
- You can now use Exhibit as an importable library, not just as a CLI program. See `recipes/exhibit_scripting.py` for examples of the basic API.
- Exhibit now correctly handles columns composed entirely out of `boolean` values. For the purposes of dataset generation they are treated as categorical rather than numerical values.

### 0.9.1 (December 5, 2021)
Hotfix a Windows-specific bug related to SQLite3 type adaptors.

### 0.9.0 (December 4, 2021)
First beta release ready for limited use in production.