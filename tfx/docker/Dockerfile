FROM tensorflow/serving
ENV MODEL_NAME=PlanetModel
ENV PORT=8501
COPY ./tfx/PlanetModel /app/tfx
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8500 --rest_api_port=$PORT \
--model_base_path=/app/tfx/PlanetModel --model_name=$MODEL_NAME \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]