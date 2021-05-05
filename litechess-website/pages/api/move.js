import Cors from 'cors'
import initMiddleware from '../../lib/init-middleware'

const cors = initMiddleware(
  Cors({
    methods: ['GET', 'POST', 'OPTIONS'],
  })
)

const axios = require('axios');

export default async (req, res) => {

  await cors(req, res)

  const fen = req.body.fen
  const options  = {
    method: 'post',
    url: `http://35.197.64.148/move`,
    headers: {
        'Content-Type': 'application/json',
    },
    data: {
      "fen" : fen
    }
  };
  
  axios(options).then((response) => {
    const moveResponse = response.data;
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.json(moveResponse);
  }).catch((error) => {
    console.log(error);
  });
}
