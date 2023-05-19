import axios from 'axios'

export async function GET(url,config){
  return (await axios.get(url,config)).data
}
