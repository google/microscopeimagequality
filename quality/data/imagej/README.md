# [ImageJ](https://imagej.net) plugin for the microscope image focus quality classifier.

## Quickstart

Assuming you already have [Apache Maven](https://maven.apache.org) installed:

```sh
mvn compile exec:java
```

## Installation in Fiji

If you have [Fiji](http://fiji.sc) installed and want to incorporate this plugin
into your installation:

```sh
# Set this to the path there Fiji.app is installed
FIJI_APP_PATH="/Users/me/Desktop/Fiji.app"
mvn -Dimagej.app.directory="${FIJI_APP_PATH}"
```

Then restart Fiji and click on the `Microscopy` menu.

## Notes

-   Instructions for installing [Apache Maven](https://maven.apache.org) might
    be as simple as `apt-get install maven` on Ubuntu and `brew install maven`
    on OS X with [homebrew](https://brew.sh)
