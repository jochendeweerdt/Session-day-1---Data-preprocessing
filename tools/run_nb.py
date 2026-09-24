import sys, nbformat, time
from nbclient import NotebookClient
for f in sys.argv[1:]:
    nb = nbformat.read(f, 4); t=time.time()
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
    print(f, "OK in", round(time.time()-t), "s")
    for i,c in enumerate(nb.cells):
        if c.cell_type=="code":
            for o in c.get("outputs",[]):
                txt = o.get("text") or o.get("data",{}).get("text/plain","")
                if o.get("output_type")=="error": print("ERROR cell",i,o.get("ename"),o.get("evalue"))
                elif txt and "--show" in sys.argv[0:1]: pass
                elif txt: print(f"[{i}]", str(txt)[:400].replace("\n","\n    "))
    nbformat.write(nb, f.replace(".ipynb", ".executed.ipynb") if "--keep" in sys.argv else "/tmp/claude-0/last_exec.ipynb")
