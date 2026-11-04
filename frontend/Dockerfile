# Do not choose the tag as latest and do not leave the tag empty as it also means latest
FROM nginx:1.28.0-alpine

# There is a specific path from where the nginx picks those files from, so the files must be transfered in the container at that location
# This path is defined by the nginx.config file, if we don't provide that file that means nginx will use it's default config that comes with the image
# That default listens on port 80, serves files from the image's default folder and returns index.html when we open host_IP:80/
# But a frontend does more than that, when the browser sends a request like /predict then nginx have to recieve that request,
# The funny thing is, it does not even know it FASTAPI exists
# When the browser sends GET / (i.e. localhost:80) the FASTAOI is not involved at all:
    # GET / -> nginx -> reads index.html from ITS OWN FOLDER (root whatever the default is) -> sends it back (so the nginx answer that itself, FastAPI never hears about it)
    # Now after getting that index.html, browser now asks for app.js or sytle.css whatever is required to run that index.html
    # So browser sends GET /app.js, now nginx looks up for <root>/app.js and sends it back and same for GET/style.css 
    # So nginx does not look anywhere else unless it's explicitely told to.
    # and remember if these files are copied somewhere else then it would not be able to find them

# Now the default.conf only tells the nginx how to serve the files (by making it look the root or default folder)
# When browser sends POST /predict that does not mean nginx somehow knows what to do with it, in the current setup FastAPI is a different container yes but even if it was in the same container then too it would not know and we would have to use localhost:8000 to hit where the FASTAPI is listening

# 1. IN the compose setup, app.js calls fetch ("/preduct", POST+image), and API_BASE is "" means same host the page came from localhost:80, but that is mapped to the frontend container so following happens
# 2. Browser: localhost-> 127.0.0.1:80 -> TCP connection -> Docker Deskops relay -> frontend container, port 80-> nginx, now nginx checks the bodysize (if image under 1MB then continue else stop with ERROR 403 entity too large or something), 
# 3. Now nginx decides what to do with /predict, the default nginx has just one rule for every path "serve files from the default folder". There's no rule mentioning /predict or the backend. So the nginx treats it like any other path, since there is no such file as predict, it returns 404 Not found. The backend container running uvicorn listening on 8000, FASTAOI has the /predict function ready but nothing reaches it.
# 4. So how to fix it? 
# 5. There are two options, (i) keep the api call within the app.js itself with the API base pointing to the backend (localhost:8000) this way the browser directly hits the localhost:8000 and gets to the uvicorn listening there and passes the request to FASTAPI and sends back the response, here nginx is not involved because frontend container does not have to hit the backend so nginx does not need to understand them.
# 6. (ii) Now if YOU WANT TO USE THE SAME ORIGIN WHERE THE REQUEST CAME FROM THEN IT MEAN THE /predict AND OTHER API CALLS WILL GO THROUGH THE FRONTEND WHICH MEANS NGINX HAS TO UNDERSTAND HOW TO HANDLE IT. IN THIS CASE WE LEAVE THE API BASE EMPTY
    # so use proxy_pass which is an instruction to nginx that means "don't answer this request yourself; forward it to this other server, then pass it's answer back", so instead of the default.conf we use the custom configuration (nginx.conf) and we use Dockerfile to replace it inside the container


# replace the default.comf with nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf 

COPY app.js index.html style.css /usr/share/nginx/html/

# if no cmd is given even then nginx will run as the base image's cmd will be used, this is what the base image has
# CMD ["nginx", "-g", "daemon off;"] daemon off means detach from terminal and run in the background




