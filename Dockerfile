
FROM pklehre/niso2019-lab3
ADD bxz858.py /bin

RUN apt-get update
RUN apt-get -y install python-numpy
RUN apt-get -y install python-sexpdata
RUN apt-get -y install python-argparse

CMD ["-username", "bxz858", "-submission", "python /bin/bxz858.py"]