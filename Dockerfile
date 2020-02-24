FROM python:3.6.9-stretch

# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.4.1 && pip install . && \
    rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
# Metric for PixCla is pure python so don't need java, nor binaries
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git && \
    cd /neubiaswg5-utilities/ && git checkout tags/v0.8.8 && pip install . && \
    rm -r /neubiaswg5-utilities

# --------------------------------------------------------------------------------------------
# Install pytorch
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# --------------------------------------------------------------------------------------------
# Install scripts and models
ADD descriptor.json /app/descriptor.json
RUN cd /app && wget http://www.montefiore.uliege.be/~rmormont/files/2020-02-24T08_06_49.613216_psp_91_0.8756.pth -O model.pth
ADD pspnet.py /app/pspnet.py
ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python", "/app/wrapper.py"]
