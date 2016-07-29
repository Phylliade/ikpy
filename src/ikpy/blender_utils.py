from ikpy import matrix_link
import sympy
import json
import subprocess
import os

def get_links_from_blender(blend_path, endpoint):
    python_script = "/usr/share/ikpy/blender_export.py"
    command = ["blender", blend_path, "--background", "--python", python_script]
    out = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    data = False
    out_data = []
    for l in out.split("\n"):
        if l == "end_data":
            data = False
        if data:
            out_data.append(l)
        if l == "begin_data":
            data = True
    json_links_list = json.loads("\n".join(out_data))
    json_links_dict = {}
    for l in json_links_list:
        json_links_dict[l["name"]] = l
    links_list = []
    while endpoint != None:
        json_link = json_links_dict[endpoint]
        if json_link["is_variable"]:
            link = matrix_link.VariableMatrixLink(json_link["name"], json_link["parent"], json_link["matrix"], [sympy.Symbol("x")])
            links_list = [link] + links_list
        else:
            link = matrix_link.ConstantMatrixLink(json_link["name"], json_link["parent"], json_link["matrix"])
            links_list = [link] + links_list
        endpoint = json_link["parent"]
    return links_list
