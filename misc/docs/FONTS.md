Fonts / 폰트

The application supports bundling a Korean font (NanumSquare) to render Korean text while keeping English text in the system font (e.g. Segoe UI).

Usage:
- Place NanumSquare TTF files into `resources/fonts/` (example: `resources/fonts/NanumSquare.ttf`).
- On startup the app will attempt to load the first `.ttf` found in that folder and apply the font selectively to Korean text.

If you build with PyInstaller, include the font file in the `datas` parameter of the `.spec` file so the font is bundled with the executable.

Example (in a .spec):

    a = Analysis(..., datas=[('resources/fonts/NanumSquare.ttf','resources/fonts')], ...)
