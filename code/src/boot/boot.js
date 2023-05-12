import {Dialog} from "quasar";

export default ({app}) => {
  app.config.errorHandler = function (err) {
    console.trace(err)
    Dialog.create({
      title:'Es ist ein Fehler aufgetreten',
      message:err.message,
      dark:true
    });
  }
}
