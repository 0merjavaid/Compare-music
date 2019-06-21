import falcon
from .request_handler import serve as server
api = falcon.API()
server = server.Serve("production/config/categories.config",
                      "production/config/attributes_index_mapping.config")
api.add_route('/attributes', server)
