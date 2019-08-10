nvidia-docker run  \
	--volume /home:/home \
	kahnchana/tf:tf1gpu \
	bash -c \
	"cd `pwd`;  ./$1"
