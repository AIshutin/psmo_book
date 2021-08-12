html:
	jupyter-book build .
	ghp-import -n -p -f _build/html

clean:
	rm -rf _build
